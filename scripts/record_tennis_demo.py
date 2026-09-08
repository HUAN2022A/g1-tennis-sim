"""离线录制网球击球演示视频（V1 站桩击球）。

复用 V1 状态机（IDLE→TRACK→ARM→SWING）与共享基础设施，但加离屏渲染：
每控制步用 MuJoCo Renderer 离屏渲染一帧，最后经 utils.video_utils 拼成 mp4。
不弹 GUI，适合无显示环境。

用法：
    python -m scripts.record_tennis_demo --track "storage/data/mocap/Tennis/p1/Random_001_Tennis 001.npz" --n_balls 4 --seed 0
    # 干净演示（固定阻力 + 零噪声 + 小可行域 → 大多命中）
    python -m scripts.record_tennis_demo --dr_k_lo 0.0125 --dr_k_hi 0.0125 --obs_sigma 0 --obs_dropout_p 0 --feas_radius 0.03
"""

from __future__ import annotations

import os
import sys

if sys.platform.startswith("linux"):
    os.environ["MUJOCO_GL"] = "egl"

from dataclasses import dataclass

import numpy as np
import mujoco
from mujoco import Renderer
from jax import tree_util as jtu

from tracking import constants as consts
from tracking.infer_utils import G1TrackInferFn, g1_infer_env_config
from tracking.policy import Args as PolicyArgs, get_policy_onnx
from utils.tennis_common import (
    build_tennis_sim, set_ball, get_ball, contact_event,
)
from utils.ball_predict import predict_intercept
from utils.ball_obs import BallObserver
from utils.tennis_clips import load_segment, reset_to_frame, calibrate
from utils.tennis_ball import (
    sample_serve, apply_ball_dr, restore_ball_physics,
    snapshot_ball_physics, step_with_drag,
)
from utils.video_utils import images_to_video

IDLE, TRACK, ARM, SWING = "IDLE", "TRACK", "ARM", "SWING"


@dataclass
class RecordArgs:
    onnx_track: str = "storage/ckpts/pns_wo_priv264.onnx"
    track: str = "storage/data/mocap/Tennis/p1/Random_001_Tennis 001.npz"
    convert_xml_path: str = str(consts.TRACK_XML)
    racket_mass: float = 0.05
    freq: int = 50
    sim_dt: float = 0.001

    n_balls: int = 4
    seed: int = 0
    ball_indices: tuple[int, ...] = ()  # 只录这些 1-based 球号（空=全部）

    # --- 发球（V1 默认）---
    feas_radius: float = 0.06
    flight_lo: float = 0.65
    flight_hi: float = 0.85
    serve_height: float = 1.0
    serve_back: float = 2.8

    # --- 球物理 DR / 观测噪声 ---
    dr_mass_lo: float = 0.95
    dr_mass_hi: float = 1.05
    dr_fric_lo: float = 0.4
    dr_fric_hi: float = 0.7
    dr_k_lo: float = 0.005
    dr_k_hi: float = 0.02
    obs_sigma: float = 0.005
    obs_dropout_p: float = 0.05
    obs_delay_frames: int = 1
    obs_window: int = 4

    # --- 状态机 ---
    arm_horizon: float = 2.0
    z_plane_offset: float = 0.0
    hold_frames: int = 30
    pred_drag_k: float = 0.0125
    # 与 V1 定稿链路对齐的触发偏置（机理见 tennis_v1_robust.py 同名注释）：
    # 录像必须复现 V1 的逐球结果，否则摸底日志选出的命中窗口在视频里会翻车。
    trigger_bias_frames: int = -2

    # --- 视频 ---
    video: str = "logs/tennis_demo_v1.mp4"
    width: int = 1280
    height: int = 720
    fps: int = 50
    azimuth: float = 90.0
    elevation: float = -15.0
    distance: float = 4.5


def _make_camera(lookat, azimuth, elevation, distance) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.asarray(lookat, dtype=np.float64)
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def run(args: RecordArgs) -> int:
    ctrl_dt = 1.0 / args.freq
    env_cfg = g1_infer_env_config(ctrl_dt=ctrl_dt)
    policy = get_policy_onnx(PolicyArgs(onnx_track=args.onnx_track))

    print(f"[REC] Loading segment: {args.track}")
    ref_traj = load_segment(args.track, args.convert_xml_path, args.freq)
    traj_len = len(ref_traj["qpos"])

    sim, site_id, ball_adr, ball_dof = build_tennis_sim(
        racket_mass=args.racket_mass, ctrl_dt=ctrl_dt, sim_dt=args.sim_dt)
    infer_fn = G1TrackInferFn(env_cfg, sim.mj_model, policy, privileged=False)

    print("[REC] Calibrating...")
    cal = calibrate(sim, infer_fn, ref_traj, site_id, ball_adr, ball_dof,
                    ctrl_dt, 0, args.hold_frames)
    t_lead = cal["t_lead"]
    p_swing = cal["p_swing"]
    onset = cal["onset"]
    peak_local = int(round(t_lead / ctrl_dt))
    z_plane = p_swing[2] + args.z_plane_offset
    swing_end = min(onset + peak_local + 30, traj_len - 1)
    ball_body_id = sim.mj_model.body("tennis_ball").id
    base_phys = snapshot_ball_physics(sim.mj_model)
    print(f"[REC] onset={onset} T_lead={t_lead:.3f}s "
          f"P_swing={np.round(p_swing, 3).tolist()}")

    sim.mj_model.vis.global_.offwidth = args.width
    sim.mj_model.vis.global_.offheight = args.height
    renderer = Renderer(sim.mj_model, height=args.height, width=args.width)
    cam = _make_camera(p_swing, args.azimuth, args.elevation, args.distance)
    frames = []

    rng = np.random.default_rng(args.seed)
    n_hit = 0
    for ball_idx in range(args.n_balls):
        serve_pos, serve_vel, T_flight, target = sample_serve(
            p_swing, rng, args.feas_radius, args.flight_lo, args.flight_hi,
            args.serve_height, args.serve_back, design_k=args.pred_drag_k)
        launch_step = max(1, args.hold_frames - int((T_flight - t_lead) * args.freq))
        swing_frames = swing_end - onset + 1
        max_ep_steps = launch_step + int(T_flight * args.freq) + swing_frames + 20

        restore_ball_physics(sim.mj_model, base_phys)
        dr = apply_ball_dr(sim.mj_model, rng,
                           k_lo=args.dr_k_lo, k_hi=args.dr_k_hi)
        observer = BallObserver(pos_sigma=args.obs_sigma,
                                dropout_p=args.obs_dropout_p,
                                delay_frames=args.obs_delay_frames,
                                smooth_window=args.obs_window,
                                seed=rng.integers(0, 2**31 - 1))

        state = reset_to_frame(sim, infer_fn, ref_traj, onset, ball_adr, ball_dof,
                               zero_root_vel=True)
        ref_hold = jtu.tree_map(lambda x: x[onset][None], ref_traj)

        mode = IDLE
        ep_step = 0
        swing_f = onset
        hit = False
        record_this = (not args.ball_indices) or ((ball_idx + 1) in set(args.ball_indices))
        while ep_step < max_ep_steps:
            if mode == IDLE and ep_step == launch_step:
                set_ball(state.mj_data, ball_adr, ball_dof, serve_pos, serve_vel)
                mujoco.mj_forward(sim.mj_model, state.mj_data)
                mode = TRACK
                observer.reset()

            if mode == IDLE:
                motor = infer_fn.infer_onnx(
                    state, {"ref_curr": ref_hold, "ref_next": ref_hold})
                state = sim.step(state, motor)

            elif mode in (TRACK, ARM):
                motor = infer_fn.infer_onnx(
                    state, {"ref_curr": ref_hold, "ref_next": ref_hold})
                state = step_with_drag(sim, state, motor, ball_dof,
                                       ball_body_id, dr["k_drag"])
                true_pos, _ = get_ball(state.mj_data, ball_adr, ball_dof)
                observer.update(true_pos, ep_step)
                if ep_step > launch_step:
                    # refine_k=False + 固定 k 预测：与 V1 定稿路径一致（V1 在
                    # σ=5mm 短窗下 7 参数拟合病态；见 §8.2 交接文档），保证
                    # 录像逐球复现 V1 摸底结果
                    obs = observer.read_fit(ep_step, ctrl_dt,
                                            k=args.pred_drag_k,
                                            min_points=args.obs_window,
                                            refine_k=False)
                    if obs is not None:
                        opos, ovel, _k_est = obs
                        inter_p, t_arr = predict_intercept(
                            opos, ovel, z_plane=z_plane, k=args.pred_drag_k)
                        if inter_p is not None:
                            if t_arr + args.trigger_bias_frames * ctrl_dt \
                                    - t_lead <= ctrl_dt * 0.5:
                                mode = SWING
                                swing_f = onset
                            elif t_arr <= t_lead + args.arm_horizon:
                                mode = ARM
                if true_pos[2] < 0.05 and mode in (TRACK, ARM):
                    mode = SWING
                    swing_f = swing_end + 1

            else:  # SWING
                if swing_f <= swing_end:
                    ref_curr = jtu.tree_map(lambda x: x[swing_f][None], ref_traj)
                    nxt = min(swing_f + 1, traj_len - 1)
                    ref_next = jtu.tree_map(lambda x: x[nxt][None], ref_traj)
                    motor = infer_fn.infer_onnx(
                        state, {"ref_curr": ref_curr, "ref_next": ref_next})
                    state = step_with_drag(sim, state, motor, ball_dof,
                                           ball_body_id, dr["k_drag"])
                    if contact_event(sim.mj_model, state.mj_data):
                        hit = True
                    swing_f += 1
                else:
                    break

            if record_this:
                renderer.update_scene(state.mj_data, cam)
                frames.append(renderer.render())
            ep_step += 1

        state.mj_data.qfrc_applied[:] = 0.0
        n_hit += int(hit)
        if record_this and frames:
            blank = np.zeros_like(frames[-1])
            for _ in range(12):
                frames.append(blank.copy())
        print(f"[REC] Ball {ball_idx + 1}/{args.n_balls}: "
              f"{'HIT' if hit else 'miss'} (T_flight={T_flight:.2f}s, "
              f"{ep_step} frames)")

    restore_ball_physics(sim.mj_model, base_phys)
    images_to_video(frames, args.video, fps=args.fps)
    print(f"[REC] Saved {len(frames)} frames -> {args.video} "
          f"({args.fps} fps, {len(frames) / args.fps:.1f}s, {n_hit}/{args.n_balls} hits)")
    return 0


def main(args: RecordArgs) -> int:
    return run(args)


if __name__ == "__main__":
    import tyro
    raise SystemExit(main(tyro.cli(RecordArgs)))
