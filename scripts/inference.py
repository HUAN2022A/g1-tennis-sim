import os
import sys

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
if sys.platform.startswith("linux"):
    os.environ["MUJOCO_GL"] = "egl"

import tyro
import numpy as np
from tqdm import tqdm
from pathlib import Path
from absl import logging
from dataclasses import dataclass

import mujoco
import mujoco.viewer
from mujoco import Renderer
from jax import tree_util as jtu
from utils.logger import LOGGER  # noqa: F401
from loop_rate_limiters import RateLimiter

from utils.video_utils import images_to_video
from utils.ref_ghost import RefGhostRenderer
from tracking import constants as consts
from tracking.convert_qpos2kpt import qpos2kpt
from tracking.policy import Args, get_policy_onnx
from tracking.infer_utils import G1TrackMjSim, G1TrackInferFn, g1_infer_env_config, apply_ema_qpos
from tracking.metrics import (
    calculate_kpt_mae_error,
    calculate_joint_tracking_error,
    calculate_max_errors,
    calculate_root_tracking_error,
    calculate_trajectory_length,
    calculate_joint_jerk,
)


@dataclass
class InferenceArgs(Args):
    onnx_track: str = "storage/ckpts/pns_wo_priv264.onnx"
    track_dir: str = "storage/test"
    privileged: bool = False
    video_path: str = None
    headless: bool = False
    num_envs: int = 1
    device: str = "cpu"
    convert: bool = False
    freq: int = 50
    convert_xml_path: str = str(consts.TRACK_XML)
    show_ref_ghost: bool = False
    # 逐帧导出目录：每条轨迹保存一个 NPZ（ref/actual qpos、obs、nn_action、motor_targets、torque、误差）
    export_dir: str = None
    # 网球场景：加载 scene_tennis.xml（右手持拍 + 网球 + 球场），racket_mass 可调
    tennis: bool = False
    racket_mass: float = 0.3


def _load_npz_with_qpos(file_path: Path) -> dict:
    data = dict(np.load(file_path, allow_pickle=True))
    if "qpos" not in data:
        if {"root_pos", "root_rot", "dof_pos"} <= data.keys():
            data["qpos"] = np.concatenate(
                [data["root_pos"], data["root_rot"], data["dof_pos"]], axis=1
            )
        elif {"joint_pos", "body_pos_w", "body_quat_w"} <= data.keys():
            data["qpos"] = np.concatenate(
                [data["body_pos_w"][:,0,:], data["body_quat_w"][:,0,:], data["joint_pos"]], axis=1
            )
    if "qpos" not in data:
        raise ValueError(f"{file_path} missing qpos (or root_pos/root_rot/dof_pos) field, cannot convert.")
    return data


def _convert_traj_to_kpt(data: dict, mj_model: mujoco.MjModel, freq_tgt: int) -> dict:
    freq_src = float(data["frequency"]) if "frequency" in data else 50
    qpos_src = np.asarray(data["qpos"], dtype=np.float32)
    return qpos2kpt(
        mj_model,
        qpos_src=qpos_src,
        freq_src=freq_src,
        freq_tgt=freq_tgt,
        interp_sec=0.0,
        end_default_sec=0.0,
        debug=False,
        foot_contact_est=False,
        height_clip_mode=None,
        video_path=None,
    )


def main(args: InferenceArgs):
    env_cfg = g1_infer_env_config(ctrl_dt = 1 / args.freq)

    data_path = Path(args.track_dir)
    if data_path.is_file():
        traj_files = [data_path]
    elif data_path.is_dir():
        traj_files = sorted(list(data_path.glob("*.npz")))
    else:
        raise ValueError(f"{data_path} not exist.")

    if not traj_files:
        raise ValueError(f"No .npz reference trajectories found under {data_path}.")

    convert_mj_model = mujoco.MjModel.from_xml_path(args.convert_xml_path)
    traj_data = []
    for file in traj_files:
        raw_data = _load_npz_with_qpos(file)
        # Apply EMA smoothing to reference qpos before any further processing
        raw_data["qpos"] = apply_ema_qpos(raw_data["qpos"])
        if args.convert:
            traj_data.append(_convert_traj_to_kpt(raw_data, convert_mj_model, args.freq))
        else:
            traj_data.append(raw_data)

    _init_qpos = traj_data[0]["qpos"][0]
    _init_qpos[:2] = 0.0
    if args.tennis:
        from utils.tennis_scene import load_tennis_model, reset_ball, get_ball_qposadr
        mj_sim = G1TrackMjSim(init_qpos=_init_qpos, headless=args.headless, ctrl_dt=env_cfg.ctrl_dt,
                              xml_path=str(consts.ROOT_PATH / "scene_tennis.xml"))
        # G1TrackMjSim 内部从 XML 加载（不含球拍），用 MjSpec 版本替换；
        # init_qpos 需扩到完整 nq（含网球 freejoint），机器人部分保持 36 维
        mj_sim.mj_model = load_tennis_model(racket_mass=args.racket_mass)
        mj_sim.mj_model.opt.timestep = 0.001
        _full_init = np.zeros(mj_sim.mj_model.nq)
        _full_init[:36] = mj_sim.init_qpos
        _full_init[get_ball_qposadr(mj_sim.mj_model) + 3] = 1.0  # 球四元数 w=1
        mj_sim.init_qpos = _full_init
        ball_adr = get_ball_qposadr(mj_sim.mj_model)
    else:
        mj_sim = G1TrackMjSim(init_qpos=_init_qpos, headless=args.headless, ctrl_dt=env_cfg.ctrl_dt)
        ball_adr = None

    if not args.onnx_track.endswith(".onnx"):
        raise ValueError(f"Unsupported onnx_track format: {args.onnx_track} (expected .onnx)")
    policy = get_policy_onnx(args)

    infer_fn = G1TrackInferFn(env_cfg, mj_sim.mj_model, policy, privileged=args.privileged)
    state = mj_sim.init_state()
    state = mj_sim.reset(state)

    ref_ghost = RefGhostRenderer(mj_sim.mj_model) if args.show_ref_ghost else None

    if not args.headless:
        viewer_cam = mj_sim.viewer.cam
        ctrl_rate = RateLimiter(frequency=args.freq, warn=False)
    if args.video_path is not None:
        viewer_cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(viewer_cam)
        viewer_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer_cam.trackbodyid = 0
        renderer = Renderer(mj_sim.mj_model, height=480, width=640)
        buf_images = []

    # Initialize tracking metrics
    all_metrics = []

    for traj_id in range(len(traj_data)):
        ref_traj = traj_data[traj_id]
        traj_len = len(traj_data[traj_id]["qpos"])

        _init_qpos = traj_data[traj_id]["qpos"][0]
        _init_qpos[:2] = 0.0
        mj_sim.init_qpos[:36] = _init_qpos
        state = mj_sim.reset(state)

        # If reference trajectory contains qvel, strictly restore initial state using qpos + qvel
        if "qvel" in ref_traj:
            _init_qvel = ref_traj["qvel"][0]
            state.mj_data.qpos[:7] = _init_qpos[:7]
            state.mj_data.qvel[:6] = _init_qvel[:6]
            mujoco.mj_forward(mj_sim.mj_model, state.mj_data)
        if ball_adr is not None:
            reset_ball(state.mj_data, mj_sim.mj_model)
        if data_path.is_file():
            file_name = data_path.name
        else:
            file_name = traj_files[traj_id].name if traj_id < len(traj_files) else f"traj_{traj_id}.npz"
        logging.info(f"Current trajectory ID: {traj_id}/{len(traj_data)}, file: {file_name}")

        # Initialize metrics for this trajectory
        traj_metrics = {
            'kpt_pos_errors': [],
            'kpt_rot_errors': [],
            'joint_pos_errors': [],
            'joint_vel_errors': [],
            'root_pos_errors': [],
            'root_vel_errors': [],
            'root_yaw_errors': [],
            'state_history': []
        }

        # Per-frame export buffers
        export = {
            'ref_qpos': [],
            'actual_qpos': [],
            'actual_qvel': [],
            'obs': [],
            'nn_action': [],
            'motor_targets': [],
            'torque': [],
            'kpt_pos_errors': [],
            'kpt_rot_errors': [],
            'joint_pos_errors': [],
            'joint_vel_errors': [],
            'root_pos_errors': [],
            'root_vel_errors': [],
            'root_yaw_errors': [],
        }

        for track_step in tqdm(range(traj_len), desc=f"Trajectory {traj_id}", leave=False):
            ref_curr = jtu.tree_map(lambda x: x[track_step][None], ref_traj)
            track_step_next = np.clip(track_step + 1, 0, traj_len - 1)
            ref_next = jtu.tree_map(lambda x: x[track_step_next][None], ref_traj)

            action = infer_fn.infer_onnx(state, {"ref_curr": ref_curr, "ref_next": ref_next})
            motor_targets = np.asarray(action).flatten()
            torque_last = state.mj_data.ctrl.copy()
            state = mj_sim.step(state, action)

            # Calculate metrics using new methods
            kpt_pos_mae, kpt_rot_mae = calculate_kpt_mae_error(state, ref_curr, ref_next, mj_sim.mj_model)
            joint_pos_mae, joint_vel_mae = calculate_joint_tracking_error(state, ref_curr)
            root_pos_err_mm, root_vel_err_mms, root_yaw_err = calculate_root_tracking_error(state, ref_curr)

            # Store metrics
            traj_metrics['kpt_pos_errors'].append(kpt_pos_mae)
            traj_metrics['kpt_rot_errors'].append(kpt_rot_mae)
            traj_metrics['joint_pos_errors'].append(joint_pos_mae)
            traj_metrics['joint_vel_errors'].append(joint_vel_mae)
            traj_metrics['root_pos_errors'].append(root_pos_err_mm)
            traj_metrics['root_vel_errors'].append(root_vel_err_mms)
            traj_metrics['root_yaw_errors'].append(root_yaw_err)

            if args.export_dir is not None:
                export['ref_qpos'].append(ref_curr["qpos"][0].copy())
                export['actual_qpos'].append(state.mj_data.qpos.copy())
                export['actual_qvel'].append(state.mj_data.qvel.copy())
                export['obs'].append(infer_fn.info["obs"][0].copy())
                export['nn_action'].append(infer_fn.info["nn_action"][0].copy())
                export['motor_targets'].append(motor_targets.copy())
                export['torque'].append(torque_last.copy())
                export['kpt_pos_errors'].append(kpt_pos_mae)
                export['kpt_rot_errors'].append(kpt_rot_mae)
                export['joint_pos_errors'].append(joint_pos_mae)
                export['joint_vel_errors'].append(joint_vel_mae)
                export['root_pos_errors'].append(root_pos_err_mm)
                export['root_vel_errors'].append(root_vel_err_mms)
                export['root_yaw_errors'].append(root_yaw_err)

            # Store state data for trajectory length calculation
            state_data = {
                'qpos': state.mj_data.qpos.copy(),
                'qvel': state.mj_data.qvel.copy(),
                'xpos': state.mj_data.xpos.copy(),
                'xmat': state.mj_data.xmat.copy()
            }
            traj_metrics['state_history'].append(state_data)

            if ref_ghost is not None:
                ref_ghost.set_qpos(ref_curr["qpos"][0])

            if not args.headless:
                viewer = getattr(mj_sim, "viewer", None)
                if viewer is not None and ref_ghost is not None:
                    ref_ghost.reset_scene(viewer.user_scn)
                    ref_ghost.add_to_scene(viewer.user_scn)
                mj_sim.view(state)
                ctrl_rate.sleep()
            if args.video_path is not None:
                viewer_cam.azimuth = 90.0
                viewer_cam.elevation = -20.0
                viewer_cam.distance = 2.0
                renderer.update_scene(state.mj_data, viewer_cam)
                if ref_ghost is not None:
                    ref_ghost.add_to_scene(renderer.scene)
                buf_images.append(renderer.render())

        # Save per-frame export for this trajectory
        if args.export_dir is not None and len(export['actual_qpos']) > 0:
            export_path = Path(args.export_dir)
            export_path.mkdir(parents=True, exist_ok=True)
            stem = Path(file_name).stem
            npz_path = export_path / f"{stem}_frames.npz"
            np.savez_compressed(
                npz_path,
                **{k: np.asarray(v) for k, v in export.items()},
                joint_names=np.array(consts.ACTION_JOINT_NAMES),
                ctrl_dt=env_cfg.ctrl_dt,
            )
            # Flat CSV of scalar per-frame signals for quick inspection
            csv_path = export_path / f"{stem}_summary.csv"
            n_frames = len(export['actual_qpos'])
            header = ("frame,kpt_pos_mae,kpt_rot_mae,joint_pos_mae,joint_vel_mae,"
                      "root_pos_err_mm,root_vel_err_mms,root_yaw_err")
            rows = np.column_stack([
                np.arange(n_frames),
                export['kpt_pos_errors'], export['kpt_rot_errors'],
                export['joint_pos_errors'], export['joint_vel_errors'],
                export['root_pos_errors'], export['root_vel_errors'],
                export['root_yaw_errors'],
            ])
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                f.write(header + "\n")
                np.savetxt(f, rows, fmt=["%d"] + ["%.6f"] * 7, delimiter=",")
            logging.info(f"    Exported {n_frames} frames -> {npz_path}")

        # Calculate final metrics for this trajectory using new method
        actual_trajectory_length = len(ref_traj["qpos"])
        traj_length_ratio, termination_step = calculate_trajectory_length(traj_metrics['state_history'], ref_traj,
                                                                          mj_sim.mj_model)
        avg_kpt_pos_error = np.mean(traj_metrics['kpt_pos_errors'])
        avg_kpt_rot_error = np.mean(traj_metrics['kpt_rot_errors'])
        avg_joint_pos_error = np.mean(traj_metrics['joint_pos_errors'])
        avg_joint_vel_error = np.mean(traj_metrics['joint_vel_errors'])
        avg_root_pos_error = np.mean(traj_metrics['root_pos_errors'])
        avg_root_vel_error = np.mean(traj_metrics['root_vel_errors'])
        avg_root_yaw_error = np.mean(traj_metrics['root_yaw_errors'])
        avg_joint_jerk = calculate_joint_jerk(traj_metrics['state_history'], env_cfg.ctrl_dt)

        # Calculate max errors
        max_errors = calculate_max_errors(traj_metrics)

        # Print trajectory statistics
        logging.info(f"  Trajectory {traj_id} completed:")
        logging.info(f"    Completion: {traj_length_ratio:.4f} ({termination_step}/{actual_trajectory_length} steps)")
        logging.info(f"    KPT Position MAE: {avg_kpt_pos_error:.6f} m (Max: {max_errors['max_kpt_pos_error']:.6f} m)")
        logging.info(f"    KPT Rotation MAE: {avg_kpt_rot_error:.6f} rad (Max: {max_errors['max_kpt_rot_error']:.6f} rad)")
        logging.info(f"    Joint Position MAE: {avg_joint_pos_error:.6f} rad (Max: {max_errors['max_joint_pos_error']:.6f} rad)")
        logging.info(f"    Joint Velocity MAE: {avg_joint_vel_error:.6f} rad/s (Max: {max_errors['max_joint_vel_error']:.6f} rad/s)")
        logging.info(f"    Root Pos Error: {avg_root_pos_error:.3f} mm (Max: {max_errors['max_root_pos_error']:.3f} mm)")
        logging.info(f"    Root Vel Error: {avg_root_vel_error:.3f} mm/s (Max: {max_errors['max_root_vel_error']:.3f} mm/s)")
        logging.info(f"    Root Yaw Error: {avg_root_yaw_error:.6f} rad (Max: {max_errors['max_root_yaw_error']:.6f} rad)")
        logging.info(f"    Joint Jerk: {avg_joint_jerk:.3f} rad/s^3")

        # Store metrics
        all_metrics.append({
            'traj_id': traj_id,
            'file_name': file_name,
            'length_ratio': traj_length_ratio,
            'kpt_pos_mae': avg_kpt_pos_error,
            'kpt_rot_mae': avg_kpt_rot_error,
            'joint_pos_mae': avg_joint_pos_error,
            'joint_vel_mae': avg_joint_vel_error,
            'root_pos_err_mm': avg_root_pos_error,
            'root_vel_err_mms': avg_root_vel_error,
            'root_yaw_err': avg_root_yaw_error,
            'joint_jerk': avg_joint_jerk,
            'max_kpt_pos_error': max_errors['max_kpt_pos_error'],
            'max_kpt_rot_error': max_errors['max_kpt_rot_error'],
            'max_joint_pos_error': max_errors['max_joint_pos_error'],
            'max_joint_vel_error': max_errors['max_joint_vel_error'],
            'max_root_pos_error': max_errors['max_root_pos_error'],
            'max_root_vel_error': max_errors['max_root_vel_error'],
            'max_root_yaw_error': max_errors['max_root_yaw_error'],
        })

    # Print overall summary
    if all_metrics:
        logging.info(f"\n=== Overall Summary ({len(all_metrics)} trajectories) ===")
        avg_length_ratio = np.mean([m['length_ratio'] for m in all_metrics])
        avg_kpt_pos = np.mean([m['kpt_pos_mae'] for m in all_metrics])
        avg_kpt_rot = np.mean([m['kpt_rot_mae'] for m in all_metrics])
        avg_joint_pos = np.mean([m['joint_pos_mae'] for m in all_metrics])
        avg_joint_vel = np.mean([m['joint_vel_mae'] for m in all_metrics])
        avg_root_pos = np.mean([m['root_pos_err_mm'] for m in all_metrics])
        avg_root_vel = np.mean([m['root_vel_err_mms'] for m in all_metrics])
        avg_root_yaw = np.mean([m['root_yaw_err'] for m in all_metrics])
        avg_joint_jerk = np.mean([m['joint_jerk'] for m in all_metrics])

        # Calculate average max errors across all trajectories
        avg_max_kpt_pos = np.mean([m['max_kpt_pos_error'] for m in all_metrics])
        avg_max_kpt_rot = np.mean([m['max_kpt_rot_error'] for m in all_metrics])
        avg_max_joint_pos = np.mean([m['max_joint_pos_error'] for m in all_metrics])
        avg_max_joint_vel = np.mean([m['max_joint_vel_error'] for m in all_metrics])
        avg_max_root_pos = np.mean([m['max_root_pos_error'] for m in all_metrics])
        avg_max_root_vel = np.mean([m['max_root_vel_error'] for m in all_metrics])
        avg_max_root_yaw = np.mean([m['max_root_yaw_error'] for m in all_metrics])

        logging.info(f"Average Trajectory Completion: {avg_length_ratio:.4f} (0-1, 1.0=completed)")
        logging.info(f"Average KPT Position MAE: {avg_kpt_pos:.6f} m (Max: {avg_max_kpt_pos:.6f} m)")
        logging.info(f"Average KPT Rotation MAE: {avg_kpt_rot:.6f} rad (Max: {avg_max_kpt_rot:.6f} rad)")
        logging.info(f"Average Joint Position MAE: {avg_joint_pos:.6f} rad (Max: {avg_max_joint_pos:.6f} rad)")
        logging.info(f"Average Joint Velocity MAE: {avg_joint_vel:.6f} rad/s (Max: {avg_max_joint_vel:.6f} rad/s)")
        logging.info(f"Average Root Pos Error: {avg_root_pos:.3f} mm (Max: {avg_max_root_pos:.3f} mm)")
        logging.info(f"Average Root Vel Error: {avg_root_vel:.3f} mm/s (Max: {avg_max_root_vel:.3f} mm/s)")
        logging.info(f"Average Root Yaw Error: {avg_root_yaw:.6f} rad (Max: {avg_max_root_yaw:.6f} rad)")
        logging.info(f"Average Joint Jerk: {avg_joint_jerk:.3f} rad/s^3")

    if args.video_path is not None:
        images_to_video(
            buf_images,
            args.video_path,
            fps=int(args.freq),
            color_format="RGB",
        )
        logging.info(f"Video saved to: {args.video_path}")

    return all_metrics


if __name__ == "__main__":
    main(tyro.cli(InferenceArgs))
