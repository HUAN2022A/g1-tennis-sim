"""V1 鲁棒命中评估：V0 状态机 + 三层扰动（发射随机化 + 球物理 DR + 观测噪声）。

按 docs/TENNIS_STATE_MACHINE_PLAN.md 第 3.1b/c 节与第 4 节 V1 验收标准：
- 随机可行域内 50 球命中率 >= 50%
- 输出失误原因分类（MISS_SWUNG / NO_SWING / OTHER）

三层扰动：
1. 发射条件随机化：拦截目标点在 P_swing 周围半径 R 的圆盘上均匀采样
   （可行域从离线拍面轨迹推出：拍面中心在峰值帧扫过 P_swing 附近，
   V0 边界数据表明拦截偏差 ~0.08 m 即 MISS、0.15 m 判 NO_SWING），
   飞行时间 U(flight_lo, flight_hi)。发射速度由抛体解反解。
2. 球物理域随机化（LATENT Table 2 球相关项）：
   - 质量 ×U(0.95, 1.05)
   - 地面弹性（几何 solref 阻尼间接控制反弹）——V1 用切向摩擦/阻尼近似
   - 切向摩擦 U(0.4, 0.7)
   - 空气阻力 k U(k_lo, k_hi)：f = -k*m*v*|v|，用 qfrc_applied 在每个
     物理子步手动施加（MuJoCo 无内建二次阻力）。预测侧用名义值 k=0
     （真机不知道真实 k，阻力误差是 V1 的误差源之一）。
3. 观测噪声（utils/ball_obs.py）：位置均匀噪声 sigma + 帧级 dropout p +
   延迟 1-2 帧 + 4 帧滑窗速度。预测输入用观测值而非真值。

标定复用 V0（T_lead / P_swing / onset 判据不变），hold 逻辑不变。

用法：
    python -m scripts.tennis_v1_robust                      # 默认 50 球
    python -m scripts.tennis_v1_robust --sweep_noise        # 三档噪声扫描
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
from utils.ball_obs import BallObserver
from utils.tennis_clips import load_segment, reset_to_frame, calibrate
from utils.tennis_ball import (
    sample_serve,
    apply_ball_dr,
    restore_ball_physics,
    snapshot_ball_physics,
    step_with_drag,
)

# 状态机状态（与 V0 一致）
IDLE, TRACK, ARM, SWING = "IDLE", "TRACK", "ARM", "SWING"

# 噪声扫描三档（LATENT 4.2 节建议起步 5mm）
NOISE_SWEEPS = (0.0025, 0.005, 0.010)


@dataclass
class V1Args:
    onnx_track: str = "storage/ckpts/pns_wo_priv264.onnx"
    track: str = r"C:\Users\huan2\Documents\具身智能\LATENT\storage\data\mocap\Tennis\p1\Random_001_Tennis 001.npz"
    convert_xml_path: str = str(consts.TRACK_XML)
    racket_mass: float = 0.05
    freq: int = 50
    sim_dt: float = 0.001

    n_balls: int = 50
    seed: int = 0

    # --- 层 1：发射条件随机化（可行域采样）---
    # 可行域 = 能被当前片段命中的范围：拍面命中圆 ~0.08m（V0 实测），
    # 采样半径必须使三层误差预算（采样/阻力/观测）总偏移落在命中圆内。
    feas_radius: float = 0.06     # 拦截目标在 P_swing 周围圆盘半径（米）
    flight_lo: float = 0.65       # 飞行时间下限（秒）
    flight_hi: float = 0.85       # 飞行时间上限（秒）
    serve_height: float = 1.0     # 发射点高于拦截目标的高度（米）
    serve_back: float = 2.8       # 发射点在拦截目标上游的距离（米，+x 侧）

    # --- 层 2：球物理 DR（LATENT Table 2 球相关项，按 A+C 决策收窄阻力范围）---
    dr_mass_lo: float = 0.95      # 质量 ×U(0.95, 1.05)
    dr_mass_hi: float = 1.05
    dr_fric_lo: float = 0.4       # 切向摩擦 U(0.4, 0.7)
    dr_fric_hi: float = 0.7
    dr_k_lo: float = 0.005        # 空气阻力系数 k U(0.005, 0.02)（仿真侧真实值）
    dr_k_hi: float = 0.02

    # --- 层 3：观测噪声 ---
    obs_sigma: float = 0.005      # 位置噪声幅度（米，均匀半宽）
    obs_dropout_p: float = 0.05   # 帧级 dropout 概率
    obs_delay_frames: int = 1     # 观测延迟（帧）
    obs_window: int = 4           # 速度滑窗长度（LATENT 4.2 节）
    sweep_noise: bool = False     # 三档噪声扫描模式（每档 n_balls 球）

    # --- 状态机参数（与 V0 相同）---
    miss_tol: float = 0.15
    arm_horizon: float = 2.0
    z_plane_offset: float = 0.0
    hold_frames: int = 30

    # NO_SWING 稳健门控：预测需剩余时间 < gate_horizon 且连续 gate_frames
    # 帧超差才判 MISS（单帧噪声尖峰拦截偏差可达 ~1m，不构成证据）
    gate_horizon: float = 0.45     # 判定 MISS 需剩余飞行时间低于此值（秒）
    gate_frames: int = 4           # 连续超差帧数

    # 预测侧空气阻力假设：k=0.0125 为 DR 范围 U(0.005,0.02) 的无偏中点
    # （去偏真实 k 的系统分量，残差不可约）；k=0 对照跑量化"球模型辨识的价值"
    pred_drag_k: float = 0.0125

    out_dir: str = "logs/tennis_v1"


# ---------------------------------------------------------------------------
# V1 单档评估
# ---------------------------------------------------------------------------

def run_v1_pass(args: V1Args, sim, infer_fn, ref_traj, cal: dict,
                obs_sigma: float, rng: np.random.Generator,
                site_id: int, ball_adr: int, ball_dof: int,
                ctrl_dt: float, log_prefix: str = "[V1]"):
    """跑一档噪声下的 n_balls 球评估。返回 results list。"""
    t_lead = cal["t_lead"]
    p_swing = cal["p_swing"]
    onset = cal["onset"]
    peak_local = int(round(t_lead / ctrl_dt))
    traj_len = len(ref_traj["qpos"])
    z_plane = p_swing[2] + args.z_plane_offset
    swing_end = min(onset + peak_local + 30, traj_len - 1)
    swing_frames = swing_end - onset + 1
    ball_body_id = sim.mj_model.body("tennis_ball").id
    base_phys = snapshot_ball_physics(sim.mj_model)

    results = []
    for ball_idx in range(args.n_balls):
        # --- 层 1：发射随机化 ---
        serve_pos, serve_vel, T_flight, target = sample_serve(
            p_swing, rng, args.feas_radius, args.flight_lo, args.flight_hi,
            args.serve_height, args.serve_back, design_k=args.pred_drag_k)
        launch_step = max(1, args.hold_frames - int((T_flight - t_lead) * args.freq))
        max_ep_steps = launch_step + int(T_flight * args.freq) + swing_frames + 20

        # --- 层 2：球物理 DR（逐球采样）---
        restore_ball_physics(sim.mj_model, base_phys)
        dr = apply_ball_dr(sim.mj_model, rng,
                           k_lo=args.dr_k_lo, k_hi=args.dr_k_hi)

        # 真值拦截点：用真实采样 k 积分（误差预算仪表——区分"球不可达"
        # 的 true_dev 与"预测偏差"；之前只记了含噪预测偏差，缺证据链）
        true_inter_p, _ = predict_intercept(serve_pos, serve_vel, z_plane=z_plane,
                                            k=dr["k_drag"])
        true_dev = float(np.linalg.norm(true_inter_p - p_swing)) \
            if true_inter_p is not None else None
        drag_dev = float(np.linalg.norm(true_inter_p - target)) \
            if true_inter_p is not None else None

        # --- 层 3：观测器 ---
        observer = BallObserver(pos_sigma=obs_sigma, dropout_p=args.obs_dropout_p,
                                delay_frames=args.obs_delay_frames,
                                smooth_window=args.obs_window,
                                seed=rng.integers(0, 2**31 - 1))

        # episode 复位（V0 逻辑）
        state = reset_to_frame(sim, infer_fn, ref_traj, onset, ball_adr, ball_dof,
                               zero_root_vel=True)
        ref_hold = jtu.tree_map(lambda x: x[onset][None], ref_traj)

        mode = IDLE
        ep_step = 0
        swing_f = onset
        t_trigger = None
        hit = False
        contact_eps = []
        pred_log = None
        no_swing_reason = None
        early_ground = False
        win_log = []
        miss_votes = 0

        while ep_step < max_ep_steps:
            if mode == IDLE and ep_step == launch_step:
                set_ball(state.mj_data, ball_adr, ball_dof, serve_pos, serve_vel)
                mujoco.mj_forward(sim.mj_model, state.mj_data)
                mode = TRACK
                # 观测器从发射帧才开始跟踪（之前球停在远处待命位，
                # 混入滑窗会污染速度估计）。此处只 reset，不 update——
                # TRACK 分支会在同一控制步喂入发射后的首个观测。
                observer.reset()

            if mode == IDLE:
                motor = infer_fn.infer_onnx(
                    state, {"ref_curr": ref_hold, "ref_next": ref_hold})
                state = sim.step(state, motor)

            elif mode in (TRACK, ARM):
                motor = infer_fn.infer_onnx(
                    state, {"ref_curr": ref_hold, "ref_next": ref_hold})
                # 阻力作用于全程（TRACK/ARM/SWING 一致——球物理不能随状态机
                # 模式切换而改变）
                state = step_with_drag(sim, state, motor, ball_dof,
                                       ball_body_id, dr["k_drag"])
                true_pos, _ = get_ball(state.mj_data, ball_adr, ball_dof)
                observer.update(true_pos, ep_step)

                now = ep_step * ctrl_dt
                if ep_step > launch_step:
                    # 决策需等滑窗填满（2 点速度估计噪声 ~2σ/dt ≈ 0.5 m/s，
                    # 会造成假 NO_SWING / 假触发）
                    obs = observer.read_fit(ep_step, ctrl_dt,
                                            k=args.pred_drag_k,
                                            min_points=args.obs_window)
                    if obs is not None:
                        opos, ovel, k_est = obs
                        inter_p, t_arr = predict_intercept(
                            opos, ovel, z_plane=z_plane, k=k_est)
                        if inter_p is not None:
                            dev_now = np.linalg.norm(inter_p - p_swing)
                            # NO_SWING 判据要稳健：预测需已收敛（剩余时间
                            # < gate_horizon）且连续 gate_frames 帧超差才判 MISS
                            # （单帧噪声尖峰可达 ~1m，不构成证据）
                            if dev_now > args.miss_tol:
                                if (t_arr < args.gate_horizon):
                                    miss_votes += 1
                                else:
                                    miss_votes = 0
                                if miss_votes >= args.gate_frames:
                                    no_swing_reason = (
                                        f"intercept dev {dev_now:.3f}m > "
                                        f"{args.miss_tol}m for "
                                        f"{miss_votes} consecutive frames")
                                    mode = SWING
                                    swing_f = swing_end + 1
                                    break
                            else:
                                miss_votes = 0
                                if t_arr - t_lead <= ctrl_dt * 0.5:
                                    t_trigger = now
                                    pred_log = (inter_p, t_arr, dev_now)
                                    mode = SWING
                                    swing_f = onset
                                elif t_arr <= t_lead + args.arm_horizon:
                                    mode = ARM
                    # 球提前落地判定（真值）：z < 0.05 且未触发
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
                    motor = infer_fn.infer_onnx(
                        state, {"ref_curr": ref_curr, "ref_next": ref_next})
                    state = step_with_drag(sim, state, motor, ball_dof,
                                           ball_body_id, dr["k_drag"])

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

        # --- 单球统计 + 失误分类 ---
        # 触发时刻误差：预测触发时刻 t_trigger + t_lead vs 真实接触时刻
        trig_err = None
        if contact_eps and t_trigger is not None:
            t_contact = contact_eps[0] * ctrl_dt
            trig_err = t_contact - (t_trigger + t_lead)
        if no_swing_reason is not None:
            outcome = "NO_SWING" if not early_ground else "EARLY_GROUND"
        elif hit:
            outcome = "HIT"
        else:
            outcome = "MISS_SWUNG"
        # 端到端物理误差：峰值窗口内 site↔ball 最小距离（补充 true_dev 不含的
        # 挥拍跟踪误差，是第 4 层预算）
        min_site_ball_dist = min(
            (float(np.linalg.norm(b - s)) for _, s, b in win_log), default=None)
        pred_inter_p = pred_log[0] if pred_log else None
        pred_dev = pred_log[2] if pred_log else None
        prediction_bias = (pred_dev - true_dev) \
            if (pred_dev is not None and true_dev is not None) else None
        bias_vec = (pred_inter_p - true_inter_p) \
            if (pred_inter_p is not None and true_inter_p is not None) else None
        results.append({
            "ball": ball_idx + 1, "outcome": outcome, "hit": hit,
            "deviation_m": pred_dev,
            "pred_dev_m": pred_dev,
            "true_target_dev_m": float(np.linalg.norm(target - p_swing)),
            "true_inter_p": true_inter_p.tolist() if true_inter_p is not None else None,
            "true_dev_m": true_dev,
            "drag_dev_m": drag_dev,
            "prediction_bias_m": prediction_bias,
            "bias_vec": bias_vec.tolist() if bias_vec is not None else None,
            "min_site_ball_dist_m": min_site_ball_dist,
            "t_trigger": t_trigger, "trigger_err_s": trig_err,
            "contact_eps": contact_eps,
            "reason": no_swing_reason,
            "dr": dr, "serve_pos": serve_pos.tolist(),
            "serve_vel": serve_vel.tolist(), "T_flight": T_flight,
            "obs_sigma": obs_sigma,
            "window_log": win_log,
        })
        print(f"{log_prefix}   [Ball {ball_idx+1:02d}/{args.n_balls}] "
              f"{outcome:12s} dev="
              f"{f'{pred_log[2]:.3f}' if pred_log else 'n/a'} "
              f"true_dev={'-' if true_dev is None else f'{true_dev:.3f}'} "
              f"k={dr['k_drag']:.3f} "
              f"trig={'-' if t_trigger is None else f'{t_trigger:.2f}'}"
              f"{f' err={trig_err:+.3f}' if trig_err is not None else ''}"
              + (f"  ({no_swing_reason})" if no_swing_reason else ""))

    restore_ball_physics(sim.mj_model, base_phys)
    return results


def summarize(results: list, label: str) -> dict:
    n = len(results)
    n_hit = sum(1 for r in results if r["hit"])
    n_miss = sum(1 for r in results if r["outcome"] == "MISS_SWUNG")
    n_noswing = sum(1 for r in results if r["outcome"] == "NO_SWING")
    n_early = sum(1 for r in results if r["outcome"] == "EARLY_GROUND")
    # MISS_SWUNG 的偏差分布（预测拦截偏差）
    miss_devs = [r["deviation_m"] for r in results
                 if r["outcome"] == "MISS_SWUNG" and r["deviation_m"] is not None]
    trig_errs = [r["trigger_err_s"] for r in results
                 if r["trigger_err_s"] is not None]
    # 误差预算三层（对 0.08m 命中圆）—— RL 立项依据
    serve_samps = [r["true_target_dev_m"] for r in results
                   if r["true_target_dev_m"] is not None]
    drag_devs = [r["drag_dev_m"] for r in results if r["drag_dev_m"] is not None]
    true_devs = [r["true_dev_m"] for r in results if r["true_dev_m"] is not None]
    biases = [r["prediction_bias_m"] for r in results
              if r["prediction_bias_m"] is not None]
    min_dists_hit = [r["min_site_ball_dist_m"] for r in results
                     if r["outcome"] == "HIT" and r["min_site_ball_dist_m"] is not None]
    min_dists_miss = [r["min_site_ball_dist_m"] for r in results
                      if r["outcome"] == "MISS_SWUNG" and r["min_site_ball_dist_m"] is not None]
    stats = {
        "label": label, "n": n, "hit": n_hit,
        "hit_rate": n_hit / n if n else 0.0,
        "miss_swung": n_miss, "no_swing": n_noswing, "early_ground": n_early,
        "miss_dev_mean": float(np.mean(miss_devs)) if miss_devs else None,
        "miss_dev_max": float(np.max(miss_devs)) if miss_devs else None,
        "trig_err_mean": float(np.mean(trig_errs)) if trig_errs else None,
        "trig_err_absmax": float(np.max(np.abs(trig_errs))) if trig_errs else None,
        "serve_sampling_mean": float(np.mean(serve_samps)) if serve_samps else None,
        "drag_dev_mean": float(np.mean(drag_devs)) if drag_devs else None,
        "true_dev_mean": float(np.mean(true_devs)) if true_devs else None,
        "true_dev_max": float(np.max(true_devs)) if true_devs else None,
        "pred_bias_mean": float(np.mean(np.abs(biases))) if biases else None,
        "min_dist_hit_mean": float(np.mean(min_dists_hit)) if min_dists_hit else None,
        "min_dist_miss_mean": float(np.mean(min_dists_miss)) if min_dists_miss else None,
    }
    _fmt = lambda x: "n/a" if x is None else f"{x:.3f}"

    print(f"\n{label} SUMMARY: HIT {n_hit}/{n} = {stats['hit_rate']:.1%} | "
          f"MISS_SWUNG {n_miss} | NO_SWING {n_noswing} | EARLY_GROUND {n_early}")
    print(f"  Error budget (hit circle 0.08m): "
          f"serve-sampling={_fmt(stats['serve_sampling_mean'])}m, "
          f"drag-mismatch={_fmt(stats['drag_dev_mean'])}m, "
          f"prediction-bias(|mean|)={_fmt(stats['pred_bias_mean'])}m")
    if true_devs:
        print(f"  true intercept dev: mean {stats['true_dev_mean']:.3f} m, "
              f"max {stats['true_dev_max']:.3f} m")
    if min_dists_hit or min_dists_miss:
        print(f"  min site-ball dist: HIT mean {_fmt(stats['min_dist_hit_mean'])} m, "
              f"MISS_SWUNG mean {_fmt(stats['min_dist_miss_mean'])} m")
    if miss_devs:
        print(f"  MISS_SWUNG pred dev: mean {stats['miss_dev_mean']:.3f} m, "
              f"max {stats['miss_dev_max']:.3f} m")
    if trig_errs:
        print(f"  trigger timing err (contact - (t_trig+T_lead)): "
              f"mean {stats['trig_err_mean']:+.4f} s, "
              f"|max| {stats['trig_err_absmax']:.4f} s")
    return stats


def run_v1(args: V1Args):
    ctrl_dt = 1.0 / args.freq
    env_cfg = g1_infer_env_config(ctrl_dt=ctrl_dt)
    policy = get_policy_onnx(PolicyArgs(onnx_track=args.onnx_track))

    print(f"[V1] Loading segment: {args.track}")
    ref_traj = load_segment(args.track, args.convert_xml_path, args.freq)
    traj_len = len(ref_traj["qpos"])
    print(f"[V1] Segment length: {traj_len} frames")

    sim, site_id, ball_adr, ball_dof = build_tennis_sim(
        racket_mass=args.racket_mass, ctrl_dt=ctrl_dt, sim_dt=args.sim_dt)
    infer_fn = G1TrackInferFn(env_cfg, sim.mj_model, policy, privileged=False)

    print("[V1] Calibrating (reuse V0 procedure)...")
    cal = calibrate(sim, infer_fn, ref_traj, site_id, ball_adr, ball_dof,
                    ctrl_dt, 0, args.hold_frames)
    print(f"[V1] onset={cal['onset']}, T_lead={cal['t_lead']:.3f}s, "
          f"P_swing={np.round(cal['p_swing'], 3).tolist()}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sigmas = NOISE_SWEEPS if args.sweep_noise else (args.obs_sigma,)
    all_stats = []
    for sigma in sigmas:
        rng = np.random.default_rng(args.seed)
        print(f"\n[V1] ====== Pass: obs_sigma={sigma*1000:.1f} mm, "
              f"dropout={args.obs_dropout_p}, delay={args.obs_delay_frames}f, "
              f"feas_radius={args.feas_radius} m ======")
        results = run_v1_pass(args, sim, infer_fn, ref_traj, cal, sigma, rng,
                              site_id, ball_adr, ball_dof, ctrl_dt)
        label = f"[sigma={sigma*1000:.1f}mm]"
        stats = summarize(results, label)
        stats["results"] = [{k: v for k, v in r.items() if k != "window_log"}
                            for r in results]
        all_stats.append(stats)

        tag = f"sigma{int(sigma*10000):04d}"
        np.savez_compressed(
            out_dir / f"v1_robust_eval_{tag}.npz",
            stats=np.array(str({k: v for k, v in stats.items()
                                if k != "results"}), dtype=object),
            results=np.array(str(stats["results"]), dtype=object),
            window_logs=np.array(
                [[(f, p.tolist(), b.tolist()) for f, p, b in r["window_log"]]
                 for r in results], dtype=object),
            t_lead=cal["t_lead"], p_swing=cal["p_swing"], onset=cal["onset"],
        )

    # 噪声-命中率曲线
    if args.sweep_noise:
        print("\n[V1] ====== Noise-to-hit-rate curve ======")
        print("  obs_sigma (mm)   hit_rate   MISS_SWUNG   NO_SWING")
        for s in all_stats:
            print(f"  {s['label']:>14}  {s['hit_rate']:>8.1%}  "
                  f"{s['miss_swung']:>11}  {s['no_swing']:>8}")

    ok = all(s["hit_rate"] >= 0.5 for s in all_stats)
    print(f"\n[V1] Acceptance (>= 50% hit rate): {'PASS' if ok else 'FAIL'}")
    print(f"[V1] Results saved to {out_dir}")
    return 0 if ok else 1


def main(args: V1Args):
    return run_v1(args)


if __name__ == "__main__":
    import tyro
    raise SystemExit(main(tyro.cli(V1Args)))
