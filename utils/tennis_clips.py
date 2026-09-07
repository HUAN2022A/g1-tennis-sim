"""网球参考片段加载 / 复位 / 标定（自 scripts/tennis_v0_stationary.py 抽取）。

供 V0/V1/V2/V3 共用：片段加载、站桩起步复位、onset 自动判据、
T_lead/P_swing 标定。calibrate 仅依赖 ref_traj/onset/site_id，
对任意片段可直接复用（V2 多片段选择的基础）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import mujoco
from jax import tree_util as jtu

from tracking.convert_qpos2kpt import qpos2kpt
from tracking.infer_utils import G1TrackMjSim, apply_ema_qpos
from utils.tennis_common import set_ball, site_state

# onset 判据阈值：site 速度低于此值视为"慢速帧"（拍面尚未进入挥拍加速段）。
# 依据 #6 Phase A 数据：Random_001 准备/预摆段 site 速度 0.7-2.4 m/s，
# 挥拍加速段 148 帧后从 1.4 → 14.3 m/s 单调上升，2.0 m/s 是干净的分界。
ONSET_SPEED_THR = 2.0


def load_segment(path: str, convert_xml_path: str, freq: int) -> dict:
    raw = dict(np.load(path, allow_pickle=True))
    if "qpos" not in raw and {"root_pos", "root_rot", "dof_pos"} <= raw.keys():
        raw["qpos"] = np.concatenate(
            [raw["root_pos"], raw["root_rot"], raw["dof_pos"]], axis=1
        )
    if "qpos" not in raw:
        raise ValueError(f"{path} missing qpos field")
    raw["qpos"] = apply_ema_qpos(raw["qpos"])
    freq_src = float(raw.get("frequency", 50))
    return qpos2kpt(
        mujoco.MjModel.from_xml_path(convert_xml_path),
        np.float32(raw["qpos"]),
        freq_src=freq_src, freq_tgt=freq,
        interp_sec=0.0, end_default_sec=0.0,
        debug=False, foot_contact_est=False,
        height_clip_mode=None, video_path=None,
    )


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
# 多片段清单与选择（V2）
# ---------------------------------------------------------------------------

@dataclass
class ClipSpec:
    """一个挥拍片段：路径 + 语义标签 + 标定结果（懒加载）。"""
    path: str
    side: str = "other"                      # "forehand" | "backhand" | "other"
    ref_traj: dict | None = None
    cal: dict | None = None
    t_lead: float | None = None
    p_swing: np.ndarray | None = None
    onset: int | None = None
    peak_local: int | None = None
    # 标注文件里的可选人工覆盖（auto-calibrate 失败时用）
    onset_override: int | None = None
    t_lead_override: float | None = None
    p_swing_override: tuple | None = None


def load_clips(label_file: str, convert_xml_path: str, freq: int) -> list[ClipSpec]:
    """从标注 JSON 加载片段清单（只解析路径/side/覆盖，不做标定）。

    标注文件 schema（相对路径相对标注文件所在目录）：
    {"clips": [{"path": "p1/Random_001.npz", "side": "forehand",
                "onset": null, "t_lead": null, "p_swing": null}]}
    """
    label_path = Path(label_file)
    data = json.loads(label_path.read_text(encoding="utf-8"))
    clips = []
    for entry in data["clips"]:
        p = Path(entry["path"])
        if not p.is_absolute():
            p = (label_path.parent / p).resolve()
        p_swing_override = entry.get("p_swing")
        clips.append(ClipSpec(
            path=str(p),
            side=entry.get("side", "other"),
            onset_override=entry.get("onset"),
            t_lead_override=entry.get("t_lead"),
            p_swing_override=tuple(p_swing_override) if p_swing_override else None,
        ))
    return clips


def calibrate_clip(sim, infer_fn, clip: ClipSpec, site_id: int, ball_adr: int,
                   ball_dof: int, ctrl_dt: float, hold_frames: int,
                   convert_xml_path: str, freq: int) -> ClipSpec:
    """加载并标定单个片段，填充 ref_traj/cal/onset/t_lead/p_swing/peak_local。

    标注文件里的 onset/t_lead/p_swing 覆盖优先于自动标定（人工可微调）。
    """
    clip.ref_traj = load_segment(clip.path, convert_xml_path, freq)
    clip.cal = calibrate(sim, infer_fn, clip.ref_traj, site_id, ball_adr, ball_dof,
                         ctrl_dt, 0, hold_frames)
    clip.onset = clip.onset_override if clip.onset_override is not None else clip.cal["onset"]
    clip.t_lead = clip.t_lead_override if clip.t_lead_override is not None else clip.cal["t_lead"]
    clip.p_swing = (np.asarray(clip.p_swing_override, dtype=np.float64)
                    if clip.p_swing_override is not None else clip.cal["p_swing"])
    clip.peak_local = int(round(clip.t_lead / ctrl_dt))
    return clip


def select_clip(inter_p: np.ndarray, body_y: float, clips: list[ClipSpec],
                forehand_dy_sign: float = -1.0) -> ClipSpec | None:
    """按预测拦截点左右侧选择正/反手片段。

    dy = inter_p.y - body_y（相对机器人根部的左右偏移）。forehand_dy_sign
    约定正手对应哪一侧（默认 -1：正手=持拍/右侧），需按标注实测定符号。
    找不到匹配 side 的片段返回 None。
    """
    dy = inter_p[1] - body_y
    side = "forehand" if forehand_dy_sign * dy < 0 else "backhand"
    for clip in clips:
        if clip.side == side:
            return clip
    return None
