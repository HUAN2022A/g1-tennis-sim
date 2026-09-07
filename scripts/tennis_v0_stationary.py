"""V0 站桩命中状态机：IDLE → TRACK → ARM → SWING，50Hz 无界面闭环评估。

按 docs/TENNIS_STATE_MACHINE_PLAN.md 第 3 节架构实现（V0 范围，方案 A 裁剪）：
- 机器人站桩：TRACK/ARM 阶段保持在"挥拍 onset 帧"的姿态（PD 锁定）；
  触发后从 onset 帧开始跟踪挥拍段（不含片段前部的准备姿势/慢速预摆）。
- 单球发射（V0 固定模式，无随机化）；球观测 = 仿真真值（无噪声）。
- 弹道预测：重力 + 可选空气阻力（utils/ball_predict.py），只在下降段拦截。
- T_lead = 挥拍 onset 帧到拍面速度峰值帧的时长（自动判据标定，不硬编码）；
  P_swing = 站桩起步回放中峰值帧的拍面中心位置。
- 状态机：
    IDLE  --发射--> TRACK（逐帧更新预测，拦截点偏差 > tol 判 MISS 不挥拍）
    TRACK --t_arrive - T_lead 临近--> ARM
    ARM   --t >= t_trigger--> SWING（从 onset 帧跟踪挥拍段）
    SWING --片段跟踪完毕--> 结束本球
- MISS 判定：拦截点与 P_swing 偏差 > 0.15 m 时不挥拍（NO_SWING）。

片段加载/复位/标定已抽取到 utils.tennis_clips，发球到 utils.tennis_ball。

用法：
    # 20 球评估（默认）
    python -m scripts.tennis_v0_stationary
    # 仅标定并打印 T_lead / P_swing / onset
    python -m scripts.tennis_v0_stationary --calibrate_only
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
from tracking.infer_utils import (
    G1TrackInferFn,
    g1_infer_env_config,
)
from tracking.policy import Args as PolicyArgs, get_policy_onnx
from utils.tennis_common import (
    build_tennis_sim,
    set_ball,
    get_ball,
    site_state,
    contact_event,
)
from utils.ball_predict import predict_intercept
from utils.tennis_clips import load_segment, reset_to_frame, calibrate
from utils.tennis_ball import design_serve

# 状态机状态
IDLE, TRACK, ARM, SWING = "IDLE", "TRACK", "ARM", "SWING"


@dataclass
class V0Args:
    onnx_track: str = "storage/ckpts/pns_wo_priv264.onnx"
    track: str = r"C:\Users\huan2\Documents\具身智能\LATENT\storage\data\mocap\Tennis\p1\Random_001_Tennis 001.npz"
    convert_xml_path: str = str(consts.TRACK_XML)
    racket_mass: float = 0.05
    freq: int = 50
    sim_dt: float = 0.001

    # --- 发射器（固定模式；serve_vel 为空时按标定结果自动设计弹道）---
    serve_pos: tuple = ()          # 空 = 自动：P_swing + (2.8, 0, 1.0)
    serve_vel: tuple = ()          # 空 = 自动：按 flight_time 抛体解
    flight_time: float = 0.0       # 0 = 自动：max(0.7, T_lead + 0.45)
    n_balls: int = 20              # 评估球数

    # --- 状态机参数 ---
    miss_tol: float = 0.15         # 拦截点与 P_swing 偏差容差（超过判 MISS 不挥拍）
    arm_horizon: float = 2.0       # TRACK→ARM 需要 t_arrive - now <= horizon
    z_plane_offset: float = 0.0    # 击球高度平面 = P_swing.z + offset
    hold_frames: int = 30          # 站桩保持帧数（标定与在线一致）

    # --- 弹道预测 ---
    drag_k: float = 0.0            # 空气阻力系数（V0 默认 0 = 纯抛体，与仿真一致）

    # --- 标定（不传则自动）---
    t_lead: float = -1.0           # <=0 时自动标定
    p_swing: tuple = ()            # 空 tuple 时自动标定
    calibrate_only: bool = False
    calib_max_steps: int = 0       # Phase A 回放帧数上限（0 = 全轨迹）

    out_dir: str = "logs/tennis_v0"


# ---------------------------------------------------------------------------
# V0 评估主循环
# ---------------------------------------------------------------------------

def run_v0(args: V0Args):
    ctrl_dt = 1.0 / args.freq
    env_cfg = g1_infer_env_config(ctrl_dt=ctrl_dt)
    policy = get_policy_onnx(PolicyArgs(onnx_track=args.onnx_track))

    print(f"[V0] Loading segment: {args.track}")
    ref_traj = load_segment(args.track, args.convert_xml_path, args.freq)
    traj_len = len(ref_traj["qpos"])
    print(f"[V0] Segment length: {traj_len} frames ({traj_len * ctrl_dt:.1f} s)")

    # --- 标定 ---
    sim, site_id, ball_adr, ball_dof = build_tennis_sim(
        racket_mass=args.racket_mass, ctrl_dt=ctrl_dt, sim_dt=args.sim_dt)
    infer_fn = G1TrackInferFn(env_cfg, sim.mj_model, policy, privileged=False)

    if args.t_lead > 0 and len(args.p_swing) == 3:
        t_lead = args.t_lead
        p_swing = np.asarray(args.p_swing, dtype=np.float64)
        onset = 0
        peak_local = int(round(t_lead / ctrl_dt))
        print(f"[V0] Manual calibration: T_lead={t_lead:.3f}s, "
              f"P_swing={np.round(p_swing, 3).tolist()} (onset=0)")
    else:
        print("[V0] Calibrating (Phase A replay → auto onset → hold+swing replay)...")
        cal = calibrate(sim, infer_fn, ref_traj, site_id, ball_adr, ball_dof,
                        ctrl_dt, args.calib_max_steps, args.hold_frames)
        t_lead = cal["t_lead"]
        p_swing = cal["p_swing"]
        onset = cal["onset"]
        peak_local = int(round(t_lead / ctrl_dt))
        print(f"[V0] Onset auto-detected: frame {onset} "
              f"(criterion: scan back from peak {cal['peak']}; last frame with site "
              f"speed < {cal['onset_thr']} m/s was {onset - 1} at "
              f"{cal['onset_speed_before']:.2f} m/s; peak speed "
              f"{cal['peak_speed']:.2f} m/s)")
        print(f"[V0] Calibrated: T_lead={t_lead:.3f}s "
              f"(onset→peak), P_swing={np.round(p_swing, 3).tolist()}")

    if args.calibrate_only:
        return 0

    # --- 发射弹道设计 ---
    T_flight = args.flight_time if args.flight_time > 0 else max(0.7, t_lead + 0.45)
    serve_pos, serve_vel = design_serve(p_swing, T_flight, args.serve_pos)
    if args.serve_vel:
        serve_vel = np.asarray(args.serve_vel, dtype=np.float64)
    z_plane = p_swing[2] + args.z_plane_offset

    # 可行性预检
    inter_p, inter_t = predict_intercept(serve_pos, serve_vel, z_plane=z_plane,
                                         k=args.drag_k)
    if inter_p is None:
        print(f"[V0] ERROR: serve never descends through z={z_plane:.2f}. Aborting.")
        return 1
    dev = np.linalg.norm(inter_p - p_swing)
    print(f"[V0] Serve: pos={np.round(serve_pos, 3).tolist()} "
          f"vel={np.round(serve_vel, 2).tolist()} "
          f"(|v|={np.linalg.norm(serve_vel):.2f} m/s, flight {T_flight:.2f}s)")
    print(f"[V0] Serve preview: intercept @ t={inter_t:.2f}s "
          f"pos={np.round(inter_p, 3).tolist()} (dev from P_swing: {dev:.3f} m, "
          f"tol {args.miss_tol})")

    # --- 逐球评估 ---
    # 发射时机设计：让触发（t_arr <= T_lead）恰好在第 hold_frames 帧发生——
    # 与标定的站桩时长一致（站桩期间机器人有缓慢漂移 ~0.01 m/帧，保持时长
    # 必须与标定相同，否则挥拍起点位姿偏移、拍面经过点漂移 ~0.35 m）。
    # launch + (T_flight - T_lead)*freq ≈ hold_frames
    launch_step = max(1, args.hold_frames - int((T_flight - t_lead) * args.freq))
    swing_end = min(onset + peak_local + 30, traj_len - 1)
    swing_frames = swing_end - onset + 1
    max_ep_steps = launch_step + int(T_flight * args.freq) + swing_frames + 20

    results = []
    for ball_idx in range(args.n_balls):
        # episode 复位（与标定步骤 2 完全一致：站桩起步，根部速度清零）
        state = reset_to_frame(sim, infer_fn, ref_traj, onset, ball_adr, ball_dof,
                               zero_root_vel=True)
        ref_hold = jtu.tree_map(lambda x: x[onset][None], ref_traj)

        mode = IDLE
        ep_step = 0
        swing_f = onset            # 当前跟踪的片段帧号
        t_trigger = None
        hit = False
        contact_eps = []
        pred_log = None
        no_swing_reason = None
        win_log = []               # 峰值窗口 ±10 帧 site/ball 逐帧记录

        while ep_step < max_ep_steps:
            if mode == IDLE and ep_step == launch_step:
                set_ball(state.mj_data, ball_adr, ball_dof, serve_pos, serve_vel)
                mujoco.mj_forward(sim.mj_model, state.mj_data)
                mode = TRACK

            if mode == IDLE:
                # 发射前站桩保持（不挥拍）：跟踪 onset 帧锁定姿态
                motor = infer_fn.infer_onnx(
                    state, {"ref_curr": ref_hold, "ref_next": ref_hold})
                state = sim.step(state, motor)

            elif mode in (TRACK, ARM):
                motor = infer_fn.infer_onnx(
                    state, {"ref_curr": ref_hold, "ref_next": ref_hold})
                state = sim.step(state, motor)

                bpos, bvel = get_ball(state.mj_data, ball_adr, ball_dof)
                now = ep_step * ctrl_dt
                if ep_step > launch_step:
                    inter_p, t_arr = predict_intercept(
                        bpos, bvel, z_plane=z_plane, k=args.drag_k)
                    # t_arr 为相对当前时刻的剩余飞行时间；
                    # 触发条件：剩余时间 <= T_lead（拍面将在 T_lead 后到峰值）
                    if inter_p is not None:
                        dev_now = np.linalg.norm(inter_p - p_swing)
                        if dev_now > args.miss_tol:
                            no_swing_reason = (f"intercept dev {dev_now:.3f}m > "
                                               f"{args.miss_tol}m")
                            mode = SWING
                            swing_f = swing_end + 1  # 不挥拍标记
                            break
                        elif t_arr - t_lead <= ctrl_dt:
                            t_trigger = now
                            pred_log = (inter_p, t_arr, dev_now)
                            mode = SWING
                            swing_f = onset
                        elif t_arr <= t_lead + args.arm_horizon:
                            mode = ARM
            else:  # SWING：从 onset 帧跟踪挥拍段
                if swing_f <= swing_end:
                    ref_curr = jtu.tree_map(lambda x: x[swing_f][None], ref_traj)
                    nxt = min(swing_f + 1, traj_len - 1)
                    ref_next = jtu.tree_map(lambda x: x[nxt][None], ref_traj)
                    motor = infer_fn.infer_onnx(
                        state, {"ref_curr": ref_curr, "ref_next": ref_next})
                    state = sim.step(state, motor)

                    # 峰值窗口 ±10 帧逐帧记录（MISS 分析用）
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
                    break  # 片段结束

            ep_step += 1

        # --- 单球统计 ---
        if no_swing_reason is not None:
            outcome = "NO_SWING"
        elif hit:
            outcome = "HIT"
        else:
            outcome = "MISS"
        dev_str = f"{pred_log[2]:.3f}" if pred_log else "n/a"
        results.append({
            "ball": ball_idx + 1, "outcome": outcome, "hit": hit,
            "deviation_m": pred_log[2] if pred_log else None,
            "t_trigger": t_trigger, "contact_eps": contact_eps,
            "reason": no_swing_reason, "window_log": win_log,
        })
        print(f"  [Ball {ball_idx+1:02d}/{args.n_balls}] {outcome:8s} "
              f"dev={dev_str} m  "
              f"t_trigger={'-' if t_trigger is None else f'{t_trigger:.2f}'}s  "
              f"contact@eps={contact_eps if contact_eps else '-'}"
              + (f"  ({no_swing_reason})" if no_swing_reason else ""))

    # --- 汇总 ---
    n_hit = sum(1 for r in results if r["hit"])
    n_noswing = sum(1 for r in results if r["outcome"] == "NO_SWING")
    hit_rate = n_hit / len(results)
    print(f"\n[V0] ====== SUMMARY ({len(results)} balls, fixed serve) ======")
    print(f"  HIT: {n_hit}  MISS(swung): {len(results) - n_hit - n_noswing}  "
          f"NO_SWING: {n_noswing}")
    print(f"  HIT rate: {n_hit}/{len(results)} = {hit_rate:.1%} "
          f"(acceptance >= 80%)")

    # MISS 球的 ±10 帧逐帧分析输出
    for r in results:
        if r["outcome"] == "MISS":
            print(f"\n  [MISS analysis] ball {r['ball']} "
                  f"(dev={r['deviation_m']:.3f} m):")
            print("    f_rel  site_pos                ball_pos               dist")
            for f_rel, spos, bpos in r["window_log"]:
                d = np.linalg.norm(bpos - spos)
                print(f"    {f_rel:4d}  {np.round(spos, 3).tolist()}  "
                      f"{np.round(bpos, 3).tolist()}  {d:.3f}")

    # --- 保存 ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "v0_stationary_eval.npz"
    summary = [{k: v for k, v in r.items() if k != "window_log"} for r in results]
    np.savez_compressed(
        out_file,
        results=np.array(str(summary), dtype=object),
        t_lead=t_lead, p_swing=p_swing, onset=onset,
        serve_pos=serve_pos, serve_vel=serve_vel,
        hit_rate=hit_rate,
        window_logs=np.array(
            [[(f, p.tolist(), b.tolist()) for f, p, b in r["window_log"]]
             for r in results], dtype=object),
    )
    print(f"\n[V0] Results saved to {out_file}")
    return 0 if hit_rate >= 0.8 else 1


def main(args: V0Args):
    return run_v0(args)


if __name__ == "__main__":
    import tyro
    raise SystemExit(main(tyro.cli(V0Args)))
