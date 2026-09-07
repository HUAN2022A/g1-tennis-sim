"""V2 正/反手片段选择状态机：在 V1 鲁棒命中之上加多片段语义选择。

按 docs/TENNIS_STATE_MACHINE_PLAN.md V2 验收：
- 每个球按预测拦截点左右侧选择正手/反手片段（select_clip）。
- 每侧 >= 20 球、混合命中率 >= 40%（多球间隔 >= 2s）。

设计决策（待真机数据标定，见文末 NOTES）：
- 发球目标盘以正/反手 P_swing 中点为心，横向跨度 reach_y 覆盖两侧，
  使球可落在任意一侧；触发时按预测拦截点相对 body_y 的符号选片段。
- 触发时序用两片段平均 T_lead（正反手挥拍时长相近），挥拍段用选中片段
  自身的 onset/peak_local/swing_end。
- forehand_dy_sign（正手对应哪一侧）是待人工标注确认的命名常量。

用法：
    python -m scripts.tennis_v2_clipselect --clip_labels storage/data/mocap/Tennis/clip_labels.json
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
from utils.tennis_common import (
    build_tennis_sim, set_ball, get_ball, site_state, contact_event,
)
from utils.ball_predict import predict_intercept
from utils.ball_obs import BallObserver
from utils.tennis_clips import (
    load_clips, calibrate_clip, select_clip, reset_to_frame,
)
from utils.tennis_ball import (
    design_serve_with_drag, apply_ball_dr, restore_ball_physics,
    snapshot_ball_physics, step_with_drag,
)

IDLE, TRACK, ARM, SWING = "IDLE", "TRACK", "ARM", "SWING"


@dataclass
class V2Args:
    onnx_track: str = "storage/ckpts/pns_wo_priv264.onnx"
    clip_labels: str = "storage/data/mocap/Tennis/clip_labels.json"
    convert_xml_path: str = str(consts.TRACK_XML)
    racket_mass: float = 0.05
    freq: int = 50
    sim_dt: float = 0.001

    n_balls: int = 50
    seed: int = 0

    # --- 层 1：发射随机化（可行域采样，目标盘跨两侧）---
    feas_radius: float = 0.06       # x 方向偏移半宽（米）
    reach_y_margin: float = 0.10    # 横向跨度在正/反手 P_swing.y 外扩的余量（米）
    flight_lo: float = 0.65
    flight_hi: float = 0.85
    serve_height: float = 1.0
    serve_back: float = 2.8

    # --- 层 2：球物理 DR ---
    dr_mass_lo: float = 0.95
    dr_mass_hi: float = 1.05
    dr_fric_lo: float = 0.4
    dr_fric_hi: float = 0.7
    dr_k_lo: float = 0.005
    dr_k_hi: float = 0.02

    # --- 层 3：观测噪声 ---
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

    # --- 片段选择 ---
    forehand_dy_sign: float = -1.0  # 正手对应拦截点 body 的哪一侧（待标注确认）

    out_dir: str = "logs/tennis_v2"


def sample_serve_v2(forehand, backhand, rng: np.random.Generator,
                    args: V2Args) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """在正/反手两侧的可行域内随机采样发射条件。

    目标盘以两片段 P_swing 中点为心，横向跨度覆盖两侧 P_swing.y（+余量），
    使球可落在任意一侧；x/z 小偏移。飞行时间量化到 0.02s 控制网格。
    """
    p_fh = forehand.p_swing
    p_bh = backhand.p_swing
    center = (p_fh + p_bh) / 2.0
    reach_y = max(abs(p_fh[1] - center[1]), abs(p_bh[1] - center[1])) \
        + args.reach_y_margin
    dy = rng.uniform(-reach_y, reach_y)
    dx = rng.uniform(-args.feas_radius, args.feas_radius)
    dz = rng.uniform(-0.02, 0.02)
    target = np.array([center[0] + dx, center[1] + dy, center[2] + dz])
    T = rng.uniform(args.flight_lo, args.flight_hi)
    grid = 0.02
    T = 0.26 + round((T - 0.26) / grid) * grid
    T = float(np.clip(T, args.flight_lo - grid, args.flight_hi + grid))
    p0, v = design_serve_with_drag(target, T, args.pred_drag_k,
                                   args.serve_height, args.serve_back)
    return p0, v, T, target


def run_v2(args: V2Args):
    ctrl_dt = 1.0 / args.freq
    env_cfg = g1_infer_env_config(ctrl_dt=ctrl_dt)
    policy = get_policy_onnx(PolicyArgs(onnx_track=args.onnx_track))
    sim, site_id, ball_adr, ball_dof = build_tennis_sim(
        racket_mass=args.racket_mass, ctrl_dt=ctrl_dt, sim_dt=args.sim_dt)
    infer_fn = G1TrackInferFn(env_cfg, sim.mj_model, policy, privileged=False)

    # --- 加载 + 标定所有片段 ---
    clips = load_clips(args.clip_labels, args.convert_xml_path, args.freq)
    for clip in clips:
        print(f"[V2] Calibrating clip side={clip.side} path={clip.path}")
        calibrate_clip(sim, infer_fn, clip, site_id, ball_adr, ball_dof,
                       ctrl_dt, args.hold_frames, args.convert_xml_path, args.freq)
        print(f"[V2]   side={clip.side:9s} onset={clip.onset} "
              f"T_lead={clip.t_lead:.3f}s P_swing={np.round(clip.p_swing, 3).tolist()} "
              f"(P_swing.y sign = {'+' if clip.p_swing[1] >= 0 else '-'})")
    by_side = {c.side: c for c in clips}
    forehand = by_side.get("forehand")
    backhand = by_side.get("backhand")
    if forehand is None or backhand is None:
        print("[V2] ERROR: need both a forehand and a backhand clip in the label file.")
        return 1

    # 触发用平均 T_lead（正反手挥拍时长相近）；z_plane 用两片段中点 z
    t_lead_trig = (forehand.t_lead + backhand.t_lead) / 2.0
    p_center = (forehand.p_swing + backhand.p_swing) / 2.0
    z_plane = p_center[2] + args.z_plane_offset
    ball_body_id = sim.mj_model.body("tennis_ball").id
    base_phys = snapshot_ball_physics(sim.mj_model)

    rng = np.random.default_rng(args.seed)
    results = []
    for ball_idx in range(args.n_balls):
        serve_pos, serve_vel, T_flight, target = sample_serve_v2(
            forehand, backhand, rng, args)
        launch_step = max(1, args.hold_frames - int((T_flight - t_lead_trig) * args.freq))

        restore_ball_physics(sim.mj_model, base_phys)
        dr = apply_ball_dr(sim.mj_model, rng,
                           k_lo=args.dr_k_lo, k_hi=args.dr_k_hi)

        observer = BallObserver(pos_sigma=args.obs_sigma,
                                dropout_p=args.obs_dropout_p,
                                delay_frames=args.obs_delay_frames,
                                smooth_window=args.obs_window,
                                seed=rng.integers(0, 2**31 - 1))

        # 复位到"准备位"：用正手片段 onset（两侧 onset 位姿接近）；选中后
        # 若为反手则 TRACK 前重定向到反手 onset（见下）。
        clip = forehand
        state = reset_to_frame(sim, infer_fn, clip.ref_traj, clip.onset,
                               ball_adr, ball_dof, zero_root_vel=True)
        ref_hold = jtu.tree_map(lambda x: x[clip.onset][None], clip.ref_traj)

        mode = IDLE
        ep_step = 0
        swing_f = clip.onset
        swing_end = min(clip.onset + clip.peak_local + 30, len(clip.ref_traj["qpos"]) - 1)
        t_trigger = None
        hit = False
        contact_eps = []
        pred_log = None
        no_swing_reason = None
        early_ground = False
        win_log = []
        miss_votes = 0
        max_ep_steps = launch_step + int(T_flight * args.freq) + 100

        while ep_step < max_ep_steps:
            if mode == IDLE and ep_step == launch_step:
                set_ball(state.mj_data, ball_adr, ball_dof, serve_pos, serve_vel)
                mujoco.mj_forward(sim.mj_model, state.mj_data)
                mode = TRACK
                observer.reset()

            if mode == IDLE:
                motor = infer_fn.infer_onnx(state, {"ref_curr": ref_hold, "ref_next": ref_hold})
                state = sim.step(state, motor)

            elif mode in (TRACK, ARM):
                motor = infer_fn.infer_onnx(state, {"ref_curr": ref_hold, "ref_next": ref_hold})
                state = step_with_drag(sim, state, motor, ball_dof, ball_body_id, dr["k_drag"])
                true_pos, _ = get_ball(state.mj_data, ball_adr, ball_dof)
                observer.update(true_pos, ep_step)

                now = ep_step * ctrl_dt
                if ep_step > launch_step:
                    obs = observer.read_fit(ep_step, ctrl_dt,
                                            k=args.pred_drag_k,
                                            min_points=args.obs_window)
                    if obs is not None:
                        opos, ovel = obs
                        inter_p, t_arr = predict_intercept(
                            opos, ovel, z_plane=z_plane, k=args.pred_drag_k)
                        if inter_p is not None:
                            # 按预测拦截点左右侧选片段（body_y 实时读）
                            body_y = float(state.mj_data.qpos[1])
                            sel = select_clip(inter_p, body_y, clips,
                                              args.forehand_dy_sign)
                            # MISS 门控：相对选中片段 P_swing 的偏差
                            p_swing_sel = sel.p_swing if sel is not None else p_center
                            dev_now = np.linalg.norm(inter_p - p_swing_sel)
                            if dev_now > args.miss_tol:
                                if t_arr < args.gate_horizon:
                                    miss_votes += 1
                                else:
                                    miss_votes = 0
                                if miss_votes >= args.gate_frames:
                                    no_swing_reason = (
                                        f"intercept dev {dev_now:.3f}m > {args.miss_tol}m")
                                    mode = SWING
                                    swing_f = swing_end + 1
                                    break
                            else:
                                miss_votes = 0
                                if t_arr - t_lead_trig <= ctrl_dt * 0.5:
                                    t_trigger = now
                                    pred_log = (inter_p, t_arr, dev_now)
                                    # 切换到选中片段（若不同于当前准备片段）
                                    if sel is not None and sel is not clip:
                                        clip = sel
                                        state = reset_to_frame(
                                            sim, infer_fn, clip.ref_traj, clip.onset,
                                            ball_adr, ball_dof, zero_root_vel=True)
                                        ref_hold = jtu.tree_map(
                                            lambda x: x[clip.onset][None], clip.ref_traj)
                                        swing_end = min(
                                            clip.onset + clip.peak_local + 30,
                                            len(clip.ref_traj["qpos"]) - 1)
                                    swing_f = clip.onset
                                    mode = SWING
                                elif t_arr <= t_lead_trig + args.arm_horizon:
                                    mode = ARM
                    if true_pos[2] < 0.05 and mode in (TRACK, ARM):
                        early_ground = True
                        no_swing_reason = "ball hit ground before trigger"
                        mode = SWING
                        swing_f = swing_end + 1
                        break

            else:  # SWING
                if swing_f <= swing_end:
                    ref_curr = jtu.tree_map(lambda x: x[swing_f][None], clip.ref_traj)
                    nxt = min(swing_f + 1, len(clip.ref_traj["qpos"]) - 1)
                    ref_next = jtu.tree_map(lambda x: x[nxt][None], clip.ref_traj)
                    motor = infer_fn.infer_onnx(state, {"ref_curr": ref_curr, "ref_next": ref_next})
                    state = step_with_drag(sim, state, motor, ball_dof, ball_body_id, dr["k_drag"])
                    f_rel = swing_f - clip.onset
                    if abs(f_rel - clip.peak_local) <= 10:
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

        trig_err = None
        if contact_eps and t_trigger is not None:
            trig_err = contact_eps[0] * ctrl_dt - (t_trigger + t_lead_trig)
        if no_swing_reason is not None:
            outcome = "NO_SWING" if not early_ground else "EARLY_GROUND"
        elif hit:
            outcome = "HIT"
        else:
            outcome = "MISS_SWUNG"
        results.append({
            "ball": ball_idx + 1, "outcome": outcome, "hit": hit,
            "side": clip.side,
            "deviation_m": pred_log[2] if pred_log else None,
            "t_trigger": t_trigger, "trigger_err_s": trig_err,
            "contact_eps": contact_eps, "reason": no_swing_reason,
            "window_log": win_log,
        })
        print(f"[V2] [Ball {ball_idx+1:02d}/{args.n_balls}] {outcome:12s} "
              f"side={clip.side:9s} "
              f"dev={f'{pred_log[2]:.3f}' if pred_log else 'n/a'}"
              + (f"  ({no_swing_reason})" if no_swing_reason else ""))

    restore_ball_physics(sim.mj_model, base_phys)

    # --- 汇总：每侧命中 + 混合命中 ---
    n_hit = sum(1 for r in results if r["hit"])
    n = len(results)
    print(f"\n[V2] ====== SUMMARY ({n} balls) ======")
    print(f"  overall HIT {n_hit}/{n} = {n_hit/n:.1%} (acceptance mixed >= 40%)")
    for side in ("forehand", "backhand"):
        side_rs = [r for r in results if r["side"] == side]
        side_hit = sum(1 for r in side_rs if r["hit"])
        print(f"  {side:9s}: {side_hit}/{len(side_rs)} = "
              f"{side_hit/len(side_rs):.1%} (acceptance >= 20 balls/side)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "v2_clipselect_eval.npz",
        results=np.array(str([{k: v for k, v in r.items() if k != 'window_log'}
                              for r in results]), dtype=object),
        hit_rate=n_hit / n if n else 0.0,
    )
    print(f"[V2] Results saved to {out_dir}")
    return 0 if n_hit / n >= 0.4 else 1


def main(args: V2Args):
    return run_v2(args)


if __name__ == "__main__":
    import tyro
    raise SystemExit(main(tyro.cli(V2Args)))
