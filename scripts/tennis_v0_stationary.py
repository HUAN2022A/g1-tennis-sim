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

onset 自动判据（可复现，V2 换片段直接复用）：
    从峰值帧向前扫描，第一个 site 速度 < ONSET_THR (2.0 m/s) 的帧的下一帧
    即为挥拍 onset——即峰值前最后一个"慢速帧"之后（球拍开始加速前）。
    判据依据记录在标定输出中。

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
from tracking.convert_qpos2kpt import qpos2kpt
from tracking.infer_utils import (
    G1TrackMjSim,
    G1TrackInferFn,
    g1_infer_env_config,
    apply_ema_qpos,
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

# 状态机状态
IDLE, TRACK, ARM, SWING = "IDLE", "TRACK", "ARM", "SWING"

# onset 判据阈值：site 速度低于此值视为"慢速帧"（拍面尚未进入挥拍加速段）。
# 依据 #6 Phase A 数据：Random_001 准备/预摆段 site 速度 0.7-2.4 m/s，
# 挥拍加速段 148 帧后从 1.4 → 14.3 m/s 单调上升，2.0 m/s 是干净的分界。
ONSET_SPEED_THR = 2.0


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
# 参考片段加载（与 tennis_hit_test.py load_traj 相同参数）
# ---------------------------------------------------------------------------

def load_segment(args: V0Args) -> dict:
    raw = dict(np.load(args.track, allow_pickle=True))
    if "qpos" not in raw and {"root_pos", "root_rot", "dof_pos"} <= raw.keys():
        raw["qpos"] = np.concatenate(
            [raw["root_pos"], raw["root_rot"], raw["dof_pos"]], axis=1
        )
    if "qpos" not in raw:
        raise ValueError(f"{args.track} missing qpos field")
    raw["qpos"] = apply_ema_qpos(raw["qpos"])
    freq_src = float(raw.get("frequency", 50))
    return qpos2kpt(
        mujoco.MjModel.from_xml_path(args.convert_xml_path),
        np.float32(raw["qpos"]),
        freq_src=freq_src, freq_tgt=args.freq,
        interp_sec=0.0, end_default_sec=0.0,
        debug=False, foot_contact_est=False,
        height_clip_mode=None, video_path=None,
    )


# ---------------------------------------------------------------------------
# 仿真复位与站桩起步回放
# ---------------------------------------------------------------------------

def reset_to_frame(sim: G1TrackMjSim, infer_fn, ref_traj: dict, frame: int,
                   ball_adr: int, ball_dof: int, zero_root_vel: bool = False):
    """把机器人复位到参考片段第 frame 帧的位姿（含根部 xy，站桩位），
    球挪到远处待命。返回 state。

    zero_root_vel=True 时把根部速度清零——站桩起步需要静止的机器人；
    参考 onset 帧的根部速度 ~1.6 m/s（挥拍前重心已在移动），不清零会导致
    站桩保持期间机器人前冲 ~0.75 m。
    """
    state = sim.init_state()
    sim.init_qpos[:36] = ref_traj["qpos"][frame].copy()
    state = sim.reset(state)
    if zero_root_vel:
        state.mj_data.qvel[:] = 0.0
    mujoco.mj_forward(sim.mj_model, state.mj_data)
    set_ball(state.mj_data, ball_adr, ball_dof, (100.0, 100.0, 5.0))
    infer_fn.info["last_action"][:] = 0
    infer_fn.info["step"] = 0
    return state


def hold_and_replay(sim: G1TrackMjSim, infer_fn, ref_traj: dict, site_id: int,
                    onset: int, end_frame: int, hold_frames: int,
                    ball_adr: int, ball_dof: int) -> np.ndarray:
    """站桩保持 hold_frames 帧后，从 onset 帧跟踪到 end_frame，记录 site 轨迹。

    标定与在线 SWING 使用同一序列，保证标定的 P_swing 是在线可达的。
    """
    state = reset_to_frame(sim, infer_fn, ref_traj, onset, ball_adr, ball_dof,
                           zero_root_vel=True)
    ref_hold = jtu.tree_map(lambda x: x[onset][None], ref_traj)
    for _ in range(hold_frames):
        motor = infer_fn.infer_onnx(
            state, {"ref_curr": ref_hold, "ref_next": ref_hold})
        state = sim.step(state, motor)

    site_traj = []
    for f in range(onset, end_frame + 1):
        ref_curr = jtu.tree_map(lambda x: x[f][None], ref_traj)
        nxt = min(f + 1, len(ref_traj["qpos"]) - 1)
        ref_next = jtu.tree_map(lambda x: x[nxt][None], ref_traj)
        motor = infer_fn.infer_onnx(state, {"ref_curr": ref_curr, "ref_next": ref_next})
        state = sim.step(state, motor)
        site_traj.append(site_state(sim.mj_model, state.mj_data, site_id)[0])
    return np.asarray(site_traj)


# ---------------------------------------------------------------------------
# 离线标定：onset（自动判据）→ T_lead / P_swing（站桩起步回放）
# ---------------------------------------------------------------------------

def detect_onset(speeds: np.ndarray, peak: int,
                 thr: float = ONSET_SPEED_THR) -> int:
    """从峰值帧向前扫描，第一个速度 < thr 的帧的下一帧 = 挥拍 onset。

    依据：挥拍加速段速度单调上升，其前方最近的慢速帧是预摆停顿点；
    onset 取该慢速帧 +1（球拍开始加速的第一帧）。
    """
    for f in range(peak - 1, -1, -1):
        if speeds[f] < thr:
            return f + 1
    return 0


def calibrate(sim: G1TrackMjSim, infer_fn, ref_traj: dict, site_id: int,
              ball_adr: int, ball_dof: int, ctrl_dt: float,
              max_steps: int, hold_frames: int,
              onset_thr: float = ONSET_SPEED_THR) -> dict:
    """标定挥拍段参数。返回 dict（含 onset 判据记录）。

    步骤：
    1. Phase A 全轨迹无球回放 → site 速度 → 峰值帧 peak、onset（自动判据）。
    2. 站桩起步回放（hold @ onset → 跟踪 onset..peak+30）→ site 轨迹
       → 局部峰值帧 p_local、P_swing、T_lead = p_local * ctrl_dt。
       （跟踪从 onset 开始，故 T_lead = onset→峰值时长。）
    """
    traj_len = len(ref_traj["qpos"])
    n_steps = min(traj_len, max_steps) if max_steps > 0 else traj_len

    # --- 步骤 1：Phase A 全轨迹回放（球在远处，根部速度按参考） ---
    state = reset_to_frame(sim, infer_fn, ref_traj, 0, ball_adr, ball_dof,
                           zero_root_vel=False)
    speeds = []
    for f in range(n_steps):
        ref_curr = jtu.tree_map(lambda x: x[f][None], ref_traj)
        nxt = min(f + 1, traj_len - 1)
        ref_next = jtu.tree_map(lambda x: x[nxt][None], ref_traj)
        motor = infer_fn.infer_onnx(state, {"ref_curr": ref_curr, "ref_next": ref_next})
        state = sim.step(state, motor)
        _, _, svel = site_state(sim.mj_model, state.mj_data, site_id)
        speeds.append(np.linalg.norm(svel))
    speeds = np.asarray(speeds)
    peak = int(np.argmax(speeds))
    onset = detect_onset(speeds, peak, thr=onset_thr)

    # --- 步骤 2：站桩起步回放 ---
    end_frame = min(peak + 30, traj_len - 1)
    site_traj = hold_and_replay(sim, infer_fn, ref_traj, site_id,
                                onset, end_frame, hold_frames, ball_adr, ball_dof)
    # 局部峰值（site_traj 索引 = 片段帧号 - onset）
    v = np.linalg.norm(np.diff(site_traj, axis=0), axis=1) / ctrl_dt
    p_local = int(np.argmax(v)) + 1
    p_swing = site_traj[p_local]
    t_lead = p_local * ctrl_dt

    return {
        "t_lead": float(t_lead),
        "p_swing": p_swing,
        "onset": int(onset),
        "peak": int(peak),
        "peak_speed": float(speeds[peak]),
        "onset_thr": float(onset_thr),
        "onset_speed_before": float(speeds[onset - 1]) if onset > 0 else 0.0,
        "hold_frames": int(hold_frames),
        "site_traj_swing": site_traj,
    }


# ---------------------------------------------------------------------------
# 发射器（固定模式，弹道按标定结果自动设计）
# ---------------------------------------------------------------------------

def design_serve(p_swing: np.ndarray, flight_time: float,
                 serve_pos: tuple = ()) -> tuple[np.ndarray, np.ndarray]:
    """设计发射位置/速度：抛体解使球在 flight_time 后到达 P_swing。

    默认发射点在 P_swing 上游 +x 侧（场地对面方向），高度 +1m。
    v = (P - p0 + 0.5*g*T^2*e_z) / T
    """
    if serve_pos:
        p0 = np.asarray(serve_pos, dtype=np.float64)
    else:
        p0 = p_swing + np.array([2.8, 0.0, 1.0])
    T = flight_time
    v = (p_swing - p0 + np.array([0.0, 0.0, 0.5 * 9.81 * T * T])) / T
    return p0, v


# ---------------------------------------------------------------------------
# V0 评估主循环
# ---------------------------------------------------------------------------

def run_v0(args: V0Args):
    ctrl_dt = 1.0 / args.freq
    env_cfg = g1_infer_env_config(ctrl_dt=ctrl_dt)
    policy = get_policy_onnx(PolicyArgs(onnx_track=args.onnx_track))

    print(f"[V0] Loading segment: {args.track}")
    ref_traj = load_segment(args)
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
