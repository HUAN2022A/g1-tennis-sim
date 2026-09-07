"""V3 走到拦截点再挥拍：WALK → (到达且时间充足) → 重定向 → TRACK → ARM → SWING。

按 docs/TENNIS_STATE_MACHINE_PLAN.md V3 验收：移动后命中率不比站桩劣化 >10pp。

关键设计（见文末 NOTES）：
- 长滞空发球（flight ~2.3-2.7s）给 walk 留出时间（V1 的 0.65-0.85s 太短走不完）。
- 导航目标不是球拦截点本身，而是"让挥拍 P_swing 落到拦截点"：
    R_reach = P_swing - onset_root；nav_target = onset_root + (inter_p - P_swing)。
- 到达后 reset_to_frame(onset, root_xy_offset=(inter_p - P_swing)[:2]) 重定向挥拍，
  之后复用 V1 的 TRACK→ARM→SWING。
- walk policy 控制腿部 12 DoF，用 KPs_walking/KDs_walking PD；到达阈值 arrive_dist。

用法：
    python -m scripts.tennis_v3_walk_intercept --track <clip.npz> --start_offset 1.0 0.0
"""

from __future__ import annotations

import os
import sys

if sys.platform.startswith("linux"):
    os.environ["MUJOCO_GL"] = "egl"

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco
from jax import tree_util as jtu

from tracking import constants as consts
from tracking.infer_utils import G1TrackInferFn, g1_infer_env_config
from tracking.policy import Args as PolicyArgs, get_policy_onnx
from utils.sim_mj import get_sensor_data
from utils.transforms_np import quat2yaw
from utils.tennis_common import (
    build_tennis_sim, set_ball, get_ball, site_state, contact_event,
)
from utils.ball_predict import predict_intercept
from utils.ball_obs import BallObserver
from utils.tennis_clips import load_segment, reset_to_frame, calibrate
from utils.tennis_ball import (
    sample_serve, apply_ball_dr, restore_ball_physics,
    snapshot_ball_physics, step_with_drag,
)
from utils.goto import GotoController
from deploy.walk_policy import WalkPolicy
from deploy.constants import KPs_walking, KDs_walking

IDLE, TRACK, ARM, SWING, WALK = "IDLE", "TRACK", "ARM", "SWING", "WALK"


@dataclass
class V3Args:
    onnx_track: str = "storage/ckpts/pns_wo_priv264.onnx"
    onnx_walk: str = "storage/ckpts/G1-Walk/07140632_G1-Walk_v2.0.0_baseline.onnx"
    track: str = r"C:\Users\huan2\Documents\具身智能\LATENT\storage\data\mocap\Tennis\p1\Random_001_Tennis 001.npz"
    convert_xml_path: str = str(consts.TRACK_XML)
    racket_mass: float = 0.05
    freq: int = 50
    sim_dt: float = 0.001

    n_balls: int = 50
    seed: int = 0

    # --- 长滞空发球（给 walk 留时间）---
    feas_radius: float = 0.06
    flight_lo: float = 2.3
    flight_hi: float = 2.7
    serve_height: float = 1.5
    serve_back: float = 4.0

    # --- 球物理 DR / 观测噪声（同 V1）---
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

    # --- 状态机参数 ---
    miss_tol: float = 0.15
    arm_horizon: float = 2.0
    z_plane_offset: float = 0.0
    hold_frames: int = 30
    gate_horizon: float = 0.45
    gate_frames: int = 4
    pred_drag_k: float = 0.0125

    # --- 步行 / 重定向 ---
    start_offset: tuple = (1.0, 0.0)   # 机器人初始相对 onset 的位移（米）
    arrive_dist: float = 0.20          # goto 到达阈值
    settle_margin: float = 0.25        # 触发前需提前到达的余量（秒）

    out_dir: str = "logs/tennis_v3"


def walk_step(sim, state, walk_policy: WalkPolicy, cmd_vel: np.ndarray):
    """一个控制步的步行（walk policy + KPs_walking/KDs_walking PD）。"""
    if sim.mj_model.nq > 36:
        jnt_qpos = state.mj_data.qpos[7:36]
        jnt_qvel = state.mj_data.qvel[6:35]
    else:
        jnt_qpos = state.mj_data.qpos[7:]
        jnt_qvel = state.mj_data.qvel[6:]
    gyro = get_sensor_data(sim.mj_model, state.mj_data, "gyro_pelvis")
    motor = walk_policy.infer(state.mj_data.qpos[3:7], gyro,
                              jnt_qpos, jnt_qvel, cmd_vel.astype(np.float32))
    for _ in range(sim.num_sim_substeps):
        torques = KPs_walking * (motor - jnt_qpos) + KDs_walking * (-jnt_qvel)
        torques = np.clip(torques, -consts.TORQUE_LIMIT, consts.TORQUE_LIMIT)
        state.mj_data.ctrl[:] = torques
        mujoco.mj_step(sim.mj_model, state.mj_data)
    return state


def run_v3(args: V3Args):
    ctrl_dt = 1.0 / args.freq
    env_cfg = g1_infer_env_config(ctrl_dt=ctrl_dt)
    policy = get_policy_onnx(PolicyArgs(onnx_track=args.onnx_track))
    walk_policy = WalkPolicy(args.onnx_walk)
    goto = GotoController(arrive_dist=args.arrive_dist)

    sim, site_id, ball_adr, ball_dof = build_tennis_sim(
        racket_mass=args.racket_mass, ctrl_dt=ctrl_dt, sim_dt=args.sim_dt)
    infer_fn = G1TrackInferFn(env_cfg, sim.mj_model, policy, privileged=False)

    ref_traj = load_segment(args.track, args.convert_xml_path, args.freq)
    cal = calibrate(sim, infer_fn, ref_traj, site_id, ball_adr, ball_dof,
                    ctrl_dt, 0, args.hold_frames)
    t_lead = cal["t_lead"]
    p_swing = cal["p_swing"]
    onset = cal["onset"]
    peak_local = int(round(t_lead / ctrl_dt))
    onset_root = ref_traj["qpos"][onset][0:2].copy()
    traj_len = len(ref_traj["qpos"])
    swing_end = min(onset + peak_local + 30, traj_len - 1)
    z_plane = p_swing[2] + args.z_plane_offset
    ball_body_id = sim.mj_model.body("tennis_ball").id
    base_phys = snapshot_ball_physics(sim.mj_model)
    print(f"[V3] onset={onset} T_lead={t_lead:.3f}s P_swing="
          f"{np.round(p_swing, 3).tolist()} onset_root={np.round(onset_root, 3).tolist()}")

    rng = np.random.default_rng(args.seed)
    results = []
    for ball_idx in range(args.n_balls):
        serve_pos, serve_vel, T_flight, target = sample_serve(
            p_swing, rng, args.feas_radius, args.flight_lo, args.flight_hi,
            args.serve_height, args.serve_back, design_k=args.pred_drag_k)
        launch_step = max(1, args.hold_frames - int((T_flight - t_lead) * args.freq))

        restore_ball_physics(sim.mj_model, base_phys)
        dr = apply_ball_dr(sim.mj_model, rng,
                           k_lo=args.dr_k_lo, k_hi=args.dr_k_hi)
        observer = BallObserver(pos_sigma=args.obs_sigma,
                                dropout_p=args.obs_dropout_p,
                                delay_frames=args.obs_delay_frames,
                                smooth_window=args.obs_window,
                                seed=rng.integers(0, 2**31 - 1))

        # 初始位移：机器人从 onset + start_offset 出发，走向拦截点
        state = reset_to_frame(sim, infer_fn, ref_traj, onset, ball_adr, ball_dof,
                               zero_root_vel=True, root_xy_offset=tuple(args.start_offset))
        ref_hold = jtu.tree_map(lambda x: x[onset][None], ref_traj)

        mode = WALK
        ep_step = 0
        swing_f = onset
        t_trigger = None
        hit = False
        contact_eps = []
        no_swing_reason = None
        early_ground = False
        win_log = []
        max_ep_steps = launch_step + int(T_flight * args.freq) + 120

        while ep_step < max_ep_steps:
            if mode == WALK and ep_step == launch_step:
                set_ball(state.mj_data, ball_adr, ball_dof, serve_pos, serve_vel)
                mujoco.mj_forward(sim.mj_model, state.mj_data)
                observer.reset()

            if mode == WALK:
                if ep_step < launch_step:
                    # 发射前原地站桩（walk policy 停）
                    state = walk_step(sim, state, walk_policy, np.zeros(3))
                else:
                    true_pos, _ = get_ball(state.mj_data, ball_adr, ball_dof)
                    observer.update(true_pos, ep_step)
                    # 球落地前一直走，未触发则 EARLY_GROUND
                    if true_pos[2] < 0.05:
                        early_ground = True
                        no_swing_reason = "ball hit ground before arrival"
                        break
                    obs = observer.read_fit(ep_step, ctrl_dt,
                                            k=args.pred_drag_k,
                                            min_points=args.obs_window)
                    if obs is not None:
                        opos, ovel = obs
                        inter_p, t_arr = predict_intercept(
                            opos, ovel, z_plane=z_plane, k=args.pred_drag_k)
                        if inter_p is not None:
                            # 导航目标：让 P_swing 落到拦截点
                            nav_target = onset_root + (inter_p - p_swing)[:2]
                            root_xy = state.mj_data.qpos[0:2]
                            root_yaw = quat2yaw(state.mj_data.qpos[3:7])
                            dist = float(np.linalg.norm(nav_target - root_xy))
                            if dist < args.arrive_dist \
                                    and (t_arr - t_lead) >= args.settle_margin:
                                # 到达且时间充足：重定向到实际拦截点，进入 TRACK
                                state = reset_to_frame(
                                    sim, infer_fn, ref_traj, onset, ball_adr, ball_dof,
                                    zero_root_vel=True,
                                    root_xy_offset=tuple((inter_p - p_swing)[:2]))
                                ref_hold = jtu.tree_map(lambda x: x[onset][None], ref_traj)
                                mode = TRACK
                            else:
                                cmd_vel = goto.cmd_vel(root_xy, root_yaw, nav_target)
                                state = walk_step(sim, state, walk_policy, cmd_vel)
                    else:
                        state = walk_step(sim, state, walk_policy, np.zeros(3))

            elif mode in (TRACK, ARM):
                motor = infer_fn.infer_onnx(state, {"ref_curr": ref_hold, "ref_next": ref_hold})
                state = step_with_drag(sim, state, motor, ball_dof, ball_body_id, dr["k_drag"])
                true_pos, _ = get_ball(state.mj_data, ball_adr, ball_dof)
                observer.update(true_pos, ep_step)
                now = ep_step * ctrl_dt
                obs = observer.read_fit(ep_step, ctrl_dt, k=args.pred_drag_k,
                                        min_points=args.obs_window)
                if obs is not None:
                    opos, ovel = obs
                    inter_p, t_arr = predict_intercept(opos, ovel, z_plane=z_plane,
                                                       k=args.pred_drag_k)
                    if inter_p is not None:
                        dev_now = np.linalg.norm(inter_p - p_swing)
                        if dev_now > args.miss_tol:
                            no_swing_reason = f"intercept dev {dev_now:.3f}m > {args.miss_tol}m"
                            mode = SWING
                            swing_f = swing_end + 1
                            break
                        elif t_arr - t_lead <= ctrl_dt * 0.5:
                            t_trigger = now
                            mode = SWING
                            swing_f = onset
                        elif t_arr <= t_lead + args.arm_horizon:
                            mode = ARM
                if true_pos[2] < 0.05 and mode in (TRACK, ARM):
                    early_ground = True
                    no_swing_reason = "ball hit ground before trigger"
                    mode = SWING
                    swing_f = swing_end + 1
                    break

            else:  # SWING
                if swing_f <= swing_end:
                    ref_curr = jtu.tree_map(lambda x: x[swing_f][None], ref_traj)
                    nxt = min(swing_f + 1, traj_len - 1)
                    ref_next = jtu.tree_map(lambda x: x[nxt][None], ref_traj)
                    motor = infer_fn.infer_onnx(state, {"ref_curr": ref_curr, "ref_next": ref_next})
                    state = step_with_drag(sim, state, motor, ball_dof, ball_body_id, dr["k_drag"])
                    f_rel = swing_f - onset
                    if abs(f_rel - peak_local) <= 10:
                        spos, _, _ = site_state(sim.mj_model, state.mj_data, site_id)
                        bpos, _ = get_ball(state.mj_data, ball_adr, ball_dof)
                        win_log.append((f_rel, spos.copy(), bpos.copy()))
                    if contact_event(sim.mj_model, state.mj_data):
                        hit = True
                        contact_eps.append(ep_step)
                    swing_f += 1
                else:
                    break

            ep_step += 1
        state.mj_data.qfrc_applied[:] = 0.0

        if no_swing_reason is not None:
            outcome = "EARLY_GROUND" if early_ground else "NO_SWING"
        elif hit:
            outcome = "HIT"
        else:
            outcome = "MISS_SWUNG"
        results.append({
            "ball": ball_idx + 1, "outcome": outcome, "hit": hit,
            "t_trigger": t_trigger, "contact_eps": contact_eps,
            "reason": no_swing_reason, "window_log": win_log,
        })
        print(f"[V3] [Ball {ball_idx+1:02d}/{args.n_balls}] {outcome:12s}"
              + (f"  ({no_swing_reason})" if no_swing_reason else ""))

    restore_ball_physics(sim.mj_model, base_phys)

    n_hit = sum(1 for r in results if r["hit"])
    n = len(results)
    print(f"\n[V3] ====== SUMMARY ({n} balls) ======")
    print(f"  HIT {n_hit}/{n} = {n_hit/n:.1%}")
    print(f"  (acceptance: no worse than 10pp below stationary V0/V1 hit rate)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "v3_walk_intercept_eval.npz",
        results=np.array(str([{k: v for k, v in r.items() if k != 'window_log'}
                              for r in results]), dtype=object),
        hit_rate=n_hit / n if n else 0.0,
    )
    print(f"[V3] Results saved to {out_dir}")
    return 0


def main(args: V3Args):
    return run_v3(args)


if __name__ == "__main__":
    import tyro
    raise SystemExit(main(tyro.cli(V3Args)))
