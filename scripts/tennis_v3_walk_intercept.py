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
from utils.tennis_clips import load_segment, reset_to_frame, calibrate, shift_ref_root
from utils.tennis_ball import (
    sample_serve, apply_ball_dr, restore_ball_physics,
    snapshot_ball_physics, step_with_drag, apply_drag,
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
    start_offset: tuple = (-0.4, 0.0)  # 机器人初始相对 onset 的位移（米）
    arrive_dist: float = 0.20          # goto 到达阈值
    redirect_lead: float = 0.90        # 重定向时机（秒）：t_arr 降到该值以下才重定向。
                                       # = (hold_frames + peak_local + 补偿帧) 时长，使
                                       # 重定向→固定站桩 hold_frames→挥拍峰值 正好对准球到达，
                                       # 同时让站桩 hold 时长与标定一致（标定陷阱）。

    # --- 触发时序偏置（单位=控制帧）---
    # 与 V1 的 trigger_bias_frames 同类旋钮（机理见 tennis_v1_robust.py 注释）：
    # V3 的"触发"= 重定向门限 t_arr<=redirect_lead。决策帧 k 的拟合状态评估在
    # 最新可用观测时刻（WALK 段步进前采样 ⇒ 标签 k = 时刻 k·dt，read_fit
    # cutoff=k-1 ⇒ 状态时刻 =(k-1)dt，比决策帧晚 1 帧）；重定向当拍不步进物理，
    # 随后固定站桩 hold_frames 帧、挥拍首步物理在 (r+31)dt ⇒ 评估→拍面峰值
    # 合计 hold+2+peak_local 帧——redirect_lead 的"+2 补偿帧"（0.90s=45f=
    # 30+13+2）恰好覆盖该 2 帧链路滞后，V3 不复现 V1 的系统性偏晚 −2f。
    # τ 仪表实测（logs/v3trig_diag.log，隔离 8 球零噪声）：bias=0 时 HIT 球
    # τ=球过面−拍面峰值 ≈ 0（−0.13~+0.02f），链路代数值 τ_conv −0.45~−0.87f，
    # 全体均值 −0.80f（残差是逐球 ±1f 的预测-真值差，非链路偏置）；完整
    # 配置 20 球 τ 均值 +0.05f（噪声主导 ±3f）。逐档实测 bias 每加 −1 平移
    # τ +1f：隔离 8 球命中 4/8(bias 0) → 1/8(−1) → 0/8(−2)——12m/s 回落速度
    # 下 1 帧=0.24m 垂直误差远超拍面容差，任何系统平移都净有害。故默认 0；
    # 旋钮保留供域变化时对心。
    trigger_bias_frames: int = 0

    out_dir: str = "logs/tennis_v3"


def walk_step(sim, state, walk_policy: WalkPolicy, cmd_vel: np.ndarray,
              ball_dof: int, ball_body_id: int, k_drag: float):
    """一个控制步的步行（walk policy + KPs_walking/KDs_walking PD）。

    与 step_with_drag 一致地给球施加空气阻力——球物理不能随状态机模式切换，
    否则 WALK 段球按纯抛体飞、发球/预测却按带阻力设计，拦截点预测整体失真。
    """
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
        apply_drag(state.mj_data, ball_dof, ball_body_id, sim.mj_model, k_drag)
        torques = KPs_walking * (motor - jnt_qpos) + KDs_walking * (-jnt_qvel)
        torques = np.clip(torques, -consts.TORQUE_LIMIT, consts.TORQUE_LIMIT)
        state.mj_data.ctrl[:] = torques
        mujoco.mj_step(sim.mj_model, state.mj_data)
    state.mj_data.qfrc_applied[:] = 0.0
    return state


def reset_to_standing(sim, infer_fn, start_xy, ball_adr, ball_dof):
    """复位到标准站立姿态（walk policy 的分布内），根在 start_xy。

    walk policy 围绕 DEFAULT_QPOS 训练；从网球 onset 姿态起步会使其观测
    严重偏离训练分布、走不稳。WALK 阶段必须从站立姿态出发，到达后再
    reset_to_frame 瞬切到挥拍 onset 姿态。
    """
    state = sim.init_state()
    sim.init_qpos[:36] = consts.DEFAULT_QPOS.copy()
    sim.init_qpos[0] = start_xy[0]
    sim.init_qpos[1] = start_xy[1]
    state = sim.reset(state)
    state.mj_data.qvel[:] = 0.0
    mujoco.mj_forward(sim.mj_model, state.mj_data)
    set_ball(state.mj_data, ball_adr, ball_dof, (100.0, 100.0, 5.0))
    infer_fn.info["last_action"][:] = 0
    infer_fn.info["step"] = 0
    return state


def _site_fk(sim, site_id, qpos_36, scratch) -> np.ndarray:
    """给定 36 维 qpos 做纯前向运动学，返回 racket_point site 世界位置。

    这是"参考拍面位置"（无跟踪误差）——用于区分"跟踪没跟上"还是"球/时序错了"。
    """
    scratch.qpos[:36] = qpos_36
    mujoco.mj_forward(sim.mj_model, scratch)
    return scratch.site_xpos[site_id].copy()


def run_v3(args: V3Args):
    ctrl_dt = 1.0 / args.freq
    env_cfg = g1_infer_env_config(ctrl_dt=ctrl_dt)
    policy = get_policy_onnx(PolicyArgs(onnx_track=args.onnx_track))
    walk_policy = WalkPolicy(args.onnx_walk)
    goto = GotoController(arrive_dist=args.arrive_dist)

    sim, site_id, ball_adr, ball_dof = build_tennis_sim(
        racket_mass=args.racket_mass, ctrl_dt=ctrl_dt, sim_dt=args.sim_dt)
    infer_fn = G1TrackInferFn(env_cfg, sim.mj_model, policy, privileged=False)
    scratch = mujoco.MjData(sim.mj_model)

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
        # 真值拦截点（用真实采样 k 积分）——区分"预测偏差"与"球真不可达"
        true_inter_p, _ = predict_intercept(serve_pos, serve_vel, z_plane=z_plane,
                                            k=dr["k_drag"])
        serve_diag = {
            "T_flight": T_flight, "launch_step": launch_step,
            "target": target.copy(), "serve_pos": serve_pos.copy(),
            "true_inter_p": true_inter_p.copy() if true_inter_p is not None else None,
            "k_true": dr["k_drag"],
        }
        observer = BallObserver(pos_sigma=args.obs_sigma,
                                dropout_p=args.obs_dropout_p,
                                delay_frames=args.obs_delay_frames,
                                smooth_window=args.obs_window,
                                seed=rng.integers(0, 2**31 - 1))

        # 初始位移：从站立姿态出发（walk 分布内），根在 onset_root + start_offset
        start_xy = onset_root + np.asarray(args.start_offset, dtype=np.float64)
        state = reset_to_standing(sim, infer_fn, start_xy, ball_adr, ball_dof)
        ref_cur = ref_traj
        ref_hold = jtu.tree_map(lambda x: x[onset][None], ref_cur)

        mode = WALK
        ep_step = 0
        swing_f = onset
        t_trigger = None
        hit = False
        contact_eps = []
        no_swing_reason = None
        early_ground = False
        win_log = []
        diag = {}
        hold_count = 0
        max_ep_steps = launch_step + int(T_flight * args.freq) + 120

        while ep_step < max_ep_steps:
            if mode == WALK and ep_step == launch_step:
                set_ball(state.mj_data, ball_adr, ball_dof, serve_pos, serve_vel)
                mujoco.mj_forward(sim.mj_model, state.mj_data)
                observer.reset()

            if mode == WALK:
                if ep_step < launch_step:
                    # 发射前原地站桩（walk policy 停）
                    state = walk_step(sim, state, walk_policy, np.zeros(3),
                                      ball_dof, ball_body_id, dr["k_drag"])
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
                        opos, ovel, k_est = obs
                        inter_p, t_arr = predict_intercept(
                            opos, ovel, z_plane=z_plane, k=k_est)
                        if inter_p is not None:
                            # 导航目标固定为 onset_root（标称拦截点）。实际拦截偏移
                            # 在重定向瞬间用已收敛的预测统一施加——否则早期预测噪声
                            # 会让 nav_target 漂移，慢速步行来不及绕回。
                            nav_target = onset_root
                            root_xy = state.mj_data.qpos[0:2]
                            root_yaw = quat2yaw(state.mj_data.qpos[3:7])
                            dist = float(np.linalg.norm(nav_target - root_xy))
                            # 触发门限加 trigger_bias_frames*ctrl_dt 平移（V1 同款
                            # 机理旋钮，默认 0=不平移，见 V3Args 注释）
                            if dist < args.arrive_dist and \
                                    t_arr + args.trigger_bias_frames * ctrl_dt \
                                    <= args.redirect_lead:
                                # 到达且时间充足：重定向挥拍到实际拦截点，进入 TRACK
                                # （保留飞行中的球：reset 会重置球 freejoint）
                                ball_qpos = state.mj_data.qpos[ball_adr:ball_adr + 7].copy()
                                ball_qvel = state.mj_data.qvel[ball_dof:ball_dof + 6].copy()
                                offset = tuple((inter_p - p_swing)[:2])
                                ref_cur = shift_ref_root(ref_traj, offset)
                                state = reset_to_frame(
                                    sim, infer_fn, ref_cur, onset, ball_adr, ball_dof,
                                    zero_root_vel=True, ball_state=(ball_qpos, ball_qvel))
                                ref_hold = jtu.tree_map(lambda x: x[onset][None], ref_cur)
                                diag = {
                                    "nav_target": nav_target.copy(),
                                    "root_before": root_xy.copy(),
                                    "offset": np.asarray(offset, dtype=np.float64).copy(),
                                    "inter_p": inter_p.copy(),
                                    "t_arr_at_redirect": float(t_arr),
                                    "root_after": state.mj_data.qpos[0:2].copy(),
                                    "ball_at_redirect": ball_qpos[0:3].copy(),
                                    "redirect_step": ep_step,
                                }
                                hold_count = 0
                                mode = TRACK
                            else:
                                cmd_vel = goto.cmd_vel(root_xy, root_yaw, nav_target)
                                state = walk_step(sim, state, walk_policy, cmd_vel,
                                                  ball_dof, ball_body_id, dr["k_drag"])
                    else:
                        state = walk_step(sim, state, walk_policy, np.zeros(3),
                                          ball_dof, ball_body_id, dr["k_drag"])

            elif mode in (TRACK, ARM):
                motor = infer_fn.infer_onnx(state, {"ref_curr": ref_hold, "ref_next": ref_hold})
                state = step_with_drag(sim, state, motor, ball_dof, ball_body_id, dr["k_drag"])
                true_pos, _ = get_ball(state.mj_data, ball_adr, ball_dof)
                observer.update(true_pos, ep_step)
                # 固定站桩 hold_frames（与标定一致），到点即挥拍——不与 t_arr
                # 耦合，避免站桩时长偏离标定导致挥拍跟踪漂移。
                hold_count += 1
                if hold_count >= args.hold_frames:
                    t_trigger = ep_step * ctrl_dt
                    mode = SWING
                    swing_f = onset
                if true_pos[2] < 0.05 and mode in (TRACK, ARM):
                    early_ground = True
                    no_swing_reason = "ball hit ground before trigger"
                    mode = SWING
                    swing_f = swing_end + 1
                    break

            else:  # SWING
                if swing_f <= swing_end:
                    ref_curr = jtu.tree_map(lambda x: x[swing_f][None], ref_cur)
                    nxt = min(swing_f + 1, traj_len - 1)
                    ref_next = jtu.tree_map(lambda x: x[nxt][None], ref_cur)
                    motor = infer_fn.infer_onnx(state, {"ref_curr": ref_curr, "ref_next": ref_next})
                    state = step_with_drag(sim, state, motor, ball_dof, ball_body_id, dr["k_drag"])
                    f_rel = swing_f - onset
                    if abs(f_rel - peak_local) <= 10:
                        spos, _, _ = site_state(sim.mj_model, state.mj_data, site_id)
                        bpos, _ = get_ball(state.mj_data, ball_adr, ball_dof)
                        win_log.append((f_rel, spos.copy(), bpos.copy()))
                    if f_rel == peak_local:
                        diag["site_at_peak"] = site_state(
                            sim.mj_model, state.mj_data, site_id)[0].copy()
                        diag["ball_at_peak"] = get_ball(
                            state.mj_data, ball_adr, ball_dof)[0].copy()
                        diag["ref_site_at_peak"] = _site_fk(
                            sim, site_id, ref_cur["qpos"][swing_f], scratch)
                    if contact_event(sim.mj_model, state.mj_data):
                        hit = True
                        contact_eps.append(ep_step)
                    swing_f += 1
                else:
                    break

            ep_step += 1
        state.mj_data.qfrc_applied[:] = 0.0

        # τ 仪表（方法同 V1 logs/trigfix_diag.log）：τ = 球过拦截面时刻 −
        # 拍面峰值时刻（帧）。win_log 在挥拍窗口逐帧记录 (f_rel, site, ball)，
        # 球 z 首次下穿 z_plane 的帧号（线性插值）− peak_local 即 τ。
        # 负 = 挥拍偏晚（球先过面）；None = 窗口内未捕捉到下穿（如触球后
        # 弹开、或偏差超出 ±10f 窗口）。
        tau = None
        for i in range(1, len(win_log)):
            f_prev, _, b_prev = win_log[i - 1]
            f_cur, _, b_cur = win_log[i]
            if b_prev[2] >= z_plane > b_cur[2]:
                frac = (b_prev[2] - z_plane) / (b_prev[2] - b_cur[2])
                tau = (f_prev - peak_local) + frac
                break

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
            "tau_frames": tau,
        })
        min_sd = min((float(np.linalg.norm(b - s)) for _, s, b in win_log), default=None)
        print(f"[V3] [Ball {ball_idx+1:02d}/{args.n_balls}] {outcome:12s} "
              f"min_site_ball={f'{min_sd:.3f}' if min_sd is not None else 'n/a'} "
              f"tau={'n/a' if tau is None else f'{tau:+.2f}f'} "
              f"t_trig={'-' if t_trigger is None else f'{t_trigger:.2f}'}"
              + (f"  ({no_swing_reason})" if no_swing_reason else ""))
        if diag:
            # 常规 τ（链路代数）：t_arr@redir − (hold+2+peak_local)，与实测 τ
            # 对照可分离"预测误差"与"链路偏置"（差值即预测的过面时刻误差）
            tau_conv = (diag["t_arr_at_redirect"] / ctrl_dt
                        - (args.hold_frames + 2 + peak_local))
            print(f"    [DIAG] tau(cross-peak_meas)="
                  f"{'n/a' if tau is None else f'{tau:+.2f}f'} "
                  f"(conv {tau_conv:+.2f}f) t_arr@redir="
                  f"{diag['t_arr_at_redirect']:.3f}s")
        if diag:
            print(f"    [DIAG] nav_target={np.round(diag['nav_target'],3).tolist()} "
                  f"root_before={np.round(diag['root_before'],3).tolist()} "
                  f"root_after={np.round(diag['root_after'],3).tolist()}")
            print(f"    [DIAG] offset={np.round(diag['offset'],3).tolist()} "
                  f"inter_p={np.round(diag['inter_p'],3).tolist()} "
                  f"t_arr@redir={diag['t_arr_at_redirect']:.3f} "
                  f"walk_steps={diag['redirect_step']} ({diag['redirect_step']*ctrl_dt:.2f}s)")
            print(f"    [DIAG] T_flight={serve_diag['T_flight']:.2f} "
                  f"launch_step={serve_diag['launch_step']} "
                  f"k_true={serve_diag['k_true']:.4f} "
                  f"target={np.round(serve_diag['target'],3).tolist()}")
            ti = serve_diag['true_inter_p']
            if ti is not None:
                print(f"    [DIAG] true_inter_p={np.round(ti,3).tolist()} "
                      f"|true-p_swing|={np.linalg.norm(ti-p_swing):.3f} "
                      f"|pred_inter_p-true|={np.linalg.norm(diag['inter_p']-ti):.3f}")
            if "site_at_peak" in diag:
                print(f"    [DIAG] p_swing={np.round(p_swing,3).tolist()}")
                print(f"    [DIAG] site@peak={np.round(diag['site_at_peak'],3).tolist()} "
                      f"ref_site@peak={np.round(diag['ref_site_at_peak'],3).tolist()}")
                print(f"    [DIAG] ball@peak={np.round(diag['ball_at_peak'],3).tolist()}")
                print(f"    [DIAG] |site-ref|={np.linalg.norm(diag['site_at_peak']-diag['ref_site_at_peak']):.3f} "
                      f"|site-ball|={np.linalg.norm(diag['site_at_peak']-diag['ball_at_peak']):.3f} "
                      f"|site-(p_swing+offset)|={np.linalg.norm(diag['site_at_peak']-(p_swing+np.r_[diag['offset'],0.0])):.3f} "
                      f"|ball-inter_p|={np.linalg.norm(diag['ball_at_peak']-diag['inter_p']):.3f}")

    restore_ball_physics(sim.mj_model, base_phys)

    n_hit = sum(1 for r in results if r["hit"])
    n = len(results)
    taus = [r["tau_frames"] for r in results if r["tau_frames"] is not None]
    print(f"\n[V3] ====== SUMMARY ({n} balls) ======")
    print(f"  HIT {n_hit}/{n} = {n_hit/n:.1%}")
    if taus:
        print(f"  tau (ball-cross-plane - racket-peak): mean {np.mean(taus):+.2f}f, "
              f"min {np.min(taus):+.2f}f, max {np.max(taus):+.2f}f, "
              f"n={len(taus)}/{n} (neg = swing late)")
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
