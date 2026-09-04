r"""固定来球命中测试：验证拍-球碰撞与反弹方向（交接文档第 10 节第 6 条）。

方法（三阶段，两次确定性回放）：
  Phase A（记录）：加载网球场景跟踪参考轨迹，把球挪到远处（无任何相互作用），
    逐帧记录 racket_point site 的世界坐标 / 速度 / 拍面法向。
  Phase B（选点）：取 site 速度峰值帧 T_hit（挥拍瞬间），球放置点 = 该帧
    racket_point 的位置（拍面中心经过点），提前 N 帧把球瞬移到该点（补偿重力下落），
    使挥动的拍面在 T_hit 附近扫到球。
  Phase C（命中）：从头重放（ONNX CPU 推理与 mj_step 均确定性，Phase A/C 的
    机器人轨迹逐帧比对校验），球在 T_place 帧被放置，逐帧记录球/拍状态与
    接触事件，分析反弹方向与拍面法向的夹角、是否穿模。

穿模对策（CLI 可选）：--sim_dt 0.0005 减小物理步长；--margin 增大球/拍
geom 的 contact margin。

用法（Windows PowerShell）：
    .\.venv\Scripts\python.exe -m scripts.tennis_hit_test `
        --track <path-to-Random_001-Tennis-001.npz> `
        --racket_mass 0.05
可选：--sim_dt 0.0005 --margin 0.002 --place_lead 2 --out_dir logs/tennis_hit_test
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
    set_ball,
    site_state,
    contact_event,
    build_tennis_sim,
)


@dataclass
class HitTestArgs:
    onnx_track: str = "storage/ckpts/pns_wo_priv264.onnx"
    track: str = r"C:\Users\huan2\Documents\具身智能\LATENT\storage\data\mocap\Tennis\p1\Random_001_Tennis 001.npz"
    convert_xml_path: str = str(consts.TRACK_XML)
    racket_mass: float = 0.05
    freq: int = 50
    sim_dt: float = 0.001          # 物理步长（穿模时可降到 0.0005）
    margin: float = 0.0            # 球/拍 contact margin（穿模时可加 0.002）
    place_lead: int = 2            # 球提前多少控制帧放置（提前越多重力下落越多）
    max_steps: int = 0             # 只跑前 N 帧（0 = 全轨迹）
    out_dir: str = "logs/tennis_hit_test"


def build_sim(args: HitTestArgs, ctrl_dt: float):
    """构建网球场景仿真（nq=43），返回 (sim, site_id, ball_qposadr, ball_dofadr)。"""
    return build_tennis_sim(
        racket_mass=args.racket_mass, ctrl_dt=ctrl_dt,
        sim_dt=args.sim_dt, margin=args.margin, headless=True,
    )


def load_traj(args: HitTestArgs, ctrl_dt: float) -> dict:
    """加载并转换参考轨迹（与 scripts/inference.py 相同的转换参数）。"""
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
# 回放：一次确定性跟踪，可在指定帧瞬移球
# ---------------------------------------------------------------------------

def run_tracking(sim: G1TrackMjSim, infer_fn, ref_traj: dict, site_id: int,
                 ball_adr: int, ball_dof: int, max_steps: int,
                 ball_start_far: bool = True,
                 place_frame: int = -1, place_pos=None, place_vel=None):
    """跑一遍跟踪。ball_start_far=True 时球被挪到远处（Phase A 记录模式）；
    place_frame>=0 时在该控制步之后把球瞬移到 place_pos（Phase C 命中模式）。

    返回逐帧记录 dict：site_pos (T,3)、site_vel (T,3)、site_normal (T,3)、
    ball_pos (T,3)、ball_vel (T,3)、contact (T,)、qpos_robot (T,36)。
    """
    ctrl_dt = sim.ctrl_dt
    state = sim.init_state()
    state = sim.reset(state)

    # 初始参考位姿（与 scripts/inference.py 一致）
    init_q = ref_traj["qpos"][0].copy()
    init_q[:2] = 0.0
    sim.init_qpos[:36] = init_q
    state = sim.reset(state)
    if "qvel" in ref_traj:
        state.mj_data.qpos[:7] = init_q[:7]
        state.mj_data.qvel[:6] = ref_traj["qvel"][0][:6]
        mujoco.mj_forward(sim.mj_model, state.mj_data)

    if ball_start_far or place_frame >= 0:
        # Phase A：球挪到远处，保证与机器人/球拍零相互作用；
        # Phase C：放置帧之前球也停在远处（(2,0,1) 处球可能被挥拍路径扫到，
        # 污染机器人轨迹），到 place_frame 才瞬移到命中点。
        set_ball(state.mj_data, ball_adr, ball_dof, (100.0, 100.0, 5.0))
    else:
        set_ball(state.mj_data, ball_adr, ball_dof, (2.0, 0.0, 1.0))

    traj_len = len(ref_traj["qpos"])
    n_steps = min(traj_len, max_steps) if max_steps > 0 else traj_len

    rec = {k: [] for k in
           ("site_pos", "site_vel", "site_normal", "ball_pos", "ball_vel",
            "contact", "qpos_robot")}

    infer_fn.info["last_action"][:] = 0
    infer_fn.info["step"] = 0

    for step in range(n_steps):
        # 命中模式：在 place_frame 步（该步推理之前）瞬移球
        if place_frame >= 0 and step == place_frame:
            set_ball(state.mj_data, ball_adr, ball_dof, place_pos, place_vel or (0, 0, 0))
            mujoco.mj_forward(sim.mj_model, state.mj_data)

        ref_curr = jtu.tree_map(lambda x: x[step][None], ref_traj)
        nxt = min(step + 1, traj_len - 1)
        ref_next = jtu.tree_map(lambda x: x[nxt][None], ref_traj)

        motor = infer_fn.infer_onnx(state, {"ref_curr": ref_curr, "ref_next": ref_next})
        state = sim.step(state, motor)

        pos, normal, svel = site_state(sim.mj_model, state.mj_data, site_id)
        rec["site_pos"].append(pos)
        rec["site_vel"].append(svel)
        rec["site_normal"].append(normal)
        rec["ball_pos"].append(state.mj_data.qpos[ball_adr:ball_adr + 3].copy())
        rec["ball_vel"].append(state.mj_data.qvel[ball_dof:ball_dof + 3].copy())
        rec["contact"].append(contact_event(sim.mj_model, state.mj_data))
        rec["qpos_robot"].append(state.mj_data.qpos[:36].copy())

    return {k: np.asarray(v) for k, v in rec.items()}


# ---------------------------------------------------------------------------
# 分析
# ---------------------------------------------------------------------------

def analyze(rec: dict, t_hit: int, ctrl_dt: float) -> dict:
    """命中分析：接触帧、反弹速度、与拍面法向的夹角、穿模检测。"""
    contact = rec["contact"]
    frames = np.where(contact)[0]
    result = {"contact_frames": frames.tolist()}

    if len(frames) == 0:
        # 无接触：检查球是否穿越拍面（穿模）
        t = t_hit
        result["tunneling"] = bool(
            t > 0 and t + 1 < len(rec["site_pos"])
            and np.linalg.norm(rec["ball_pos"][t] - rec["site_pos"][t]) < 0.1
            and np.linalg.norm(rec["ball_vel"][t]) < 1e-6
        )
        return result

    t0 = int(frames[0])
    t_pre = max(t0 - 1, 0)
    t_post = min(t0 + 1, len(rec["ball_vel"]) - 1)
    v_pre = rec["ball_vel"][t_pre]
    v_post = rec["ball_vel"][t_post]
    n = rec["site_normal"][t0]
    n_unit = n / (np.linalg.norm(n) + 1e-9)

    result.update({
        "first_contact_frame": t0,
        "t_hit_selected": t_hit,
        "ball_vel_pre": v_pre.tolist(),
        "ball_vel_post": v_post.tolist(),
        "racket_vel_at_contact": rec["site_vel"][t0].tolist(),
        "face_normal_at_contact": n_unit.tolist(),
        "rebound_speed_mps": float(np.linalg.norm(v_post)),
        "rebound_angle_vs_normal_deg": float(np.degrees(
            np.arccos(np.clip(
                np.dot(v_post / (np.linalg.norm(v_post) + 1e-9), n_unit), -1, 1
            ))
        )),
        "racket_normal_speed_mps": float(np.dot(rec["site_vel"][t0], n_unit)),
    })

    # 穿模判定：接触发生但反弹速度异常小（球被"吞"而非弹出）
    result["tunneling"] = bool(
        np.linalg.norm(v_post) < 0.5 * abs(result["racket_normal_speed_mps"])
    )
    return result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(args: HitTestArgs):
    ctrl_dt = 1.0 / args.freq
    env_cfg = g1_infer_env_config(ctrl_dt=ctrl_dt)
    policy = get_policy_onnx(PolicyArgs(onnx_track=args.onnx_track))

    print(f"[HitTest] Loading trajectory: {args.track}")
    ref_traj = load_traj(args, ctrl_dt)
    traj_len = len(ref_traj["qpos"])
    print(f"[HitTest] Trajectory length: {traj_len} frames "
          f"({traj_len * ctrl_dt:.1f} s)")

    # ---------------- Phase A：记录拍面轨迹（球在远处） ----------------
    print("[HitTest] Phase A: recording racket_point trajectory (ball parked far away)...")
    sim_a, site_id, ball_adr, ball_dof = build_sim(args, ctrl_dt)
    infer_a = G1TrackInferFn(env_cfg, sim_a.mj_model, policy, privileged=False)
    rec_a = run_tracking(sim_a, infer_a, ref_traj, site_id, ball_adr, ball_dof,
                         args.max_steps, ball_start_far=True)

    speeds = np.linalg.norm(rec_a["site_vel"], axis=1)
    t_hit = int(np.argmax(speeds))
    print(f"[HitTest] Swing peak: frame {t_hit}/{len(speeds)} "
          f"(t={t_hit * ctrl_dt:.2f} s), racket_point speed = {speeds[t_hit]:.2f} m/s")

    # ---------------- Phase B：选择球放置点 ----------------
    # 球放在 T_hit 帧拍面中心将要经过的位置，提前 place_lead 帧放置，
    # 并补偿重力下落（静止释放，挥拍扫到球）。
    lead = max(args.place_lead, 0)
    t_place = max(t_hit - lead, 0)
    dt_fall = (t_hit - t_place) * ctrl_dt
    drop = 0.5 * 9.81 * dt_fall ** 2
    place_pos = rec_a["site_pos"][t_hit].copy()
    place_pos[2] += drop  # 补偿提前放置期间的重力下落
    print(f"[HitTest] Phase B: ball placed at frame {t_place}, "
          f"pos={np.round(place_pos, 3).tolist()} "
          f"(target contact point @ frame {t_hit}, "
          f"gravity drop compensated {drop * 1000:.1f} mm)")

    # ---------------- Phase C：命中回放 ----------------
    print("[HitTest] Phase C: replay with ball teleported to the contact point...")
    sim_c, site_id_c, ball_adr_c, ball_dof_c = build_sim(args, ctrl_dt)
    infer_c = G1TrackInferFn(env_cfg, sim_c.mj_model, policy, privileged=False)
    rec_c = run_tracking(sim_c, infer_c, ref_traj, site_id_c, ball_adr_c, ball_dof_c,
                         args.max_steps, ball_start_far=False,
                         place_frame=t_place, place_pos=place_pos)

    # 确定性校验：两阶段机器人轨迹应逐帧一致（球放置前无相互作用）
    n_cmp = min(len(rec_a["qpos_robot"]), len(rec_c["qpos_robot"]))
    qpos_diff = np.abs(rec_a["qpos_robot"][:n_cmp] - rec_c["qpos_robot"][:n_cmp])
    # 只比较放置帧之前（之后接触力会改变机器人状态——这正是要观察的）
    n_pre = min(t_place + 1, n_cmp)
    pre_diff = float(np.max(qpos_diff[:n_pre])) if n_pre > 0 else 0.0
    print(f"[HitTest] Determinism check (frames <= {t_place}): "
          f"max |qpos diff| = {pre_diff:.2e}")
    if pre_diff > 1e-3:
        print("[HitTest] WARN: replay is not deterministic; hit point may be off.")

    analysis = analyze(rec_c, t_hit, ctrl_dt)

    # ---------------- 结果输出 ----------------
    print("\n[HitTest] ====== HIT TEST SUMMARY ======")
    print(f"  Racket mass: {args.racket_mass} kg | sim_dt: {args.sim_dt} | margin: {args.margin}")
    print(f"  Swing peak frame {t_hit}: site speed {speeds[t_hit]:.2f} m/s, "
          f"pos {np.round(rec_a['site_pos'][t_hit], 3).tolist()}")
    if "first_contact_frame" in analysis:
        t0 = analysis["first_contact_frame"]
        print(f"  Contact: YES at frame {t0} (selected t_hit={t_hit}, offset {t0 - t_hit:+d})")
        print(f"  Ball velocity pre-hit:  {np.round(analysis['ball_vel_pre'], 2).tolist()} m/s")
        print(f"  Ball velocity post-hit: {np.round(analysis['ball_vel_post'], 2).tolist()} m/s "
              f"(|v|={analysis['rebound_speed_mps']:.2f} m/s)")
        print(f"  Racket velocity at contact: {np.round(analysis['racket_vel_at_contact'], 2).tolist()} m/s "
              f"(normal comp {analysis['racket_normal_speed_mps']:.2f} m/s)")
        print(f"  Face normal at contact: {np.round(analysis['face_normal_at_contact'], 3).tolist()}")
        print(f"  Rebound angle vs face normal: {analysis['rebound_angle_vs_normal_deg']:.1f} deg")
    else:
        print(f"  Contact: NO (contact frames: {analysis['contact_frames']})")
    print(f"  Tunneling suspected: {analysis['tunneling']}")
    print("======================================\n")

    # 保存逐帧数据
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.track).stem.replace(" ", "_")
    tag = f"dt{args.sim_dt}_m{args.margin}_lead{args.place_lead}"
    out_file = out_dir / f"hit_test_{stem}_{tag}.npz"
    np.savez_compressed(
        out_file,
        site_pos=rec_a["site_pos"], site_vel=rec_a["site_vel"],
        site_normal=rec_a["site_normal"], site_speeds=speeds,
        t_hit=t_hit, t_place=t_place, place_pos=place_pos,
        phase_c_site_pos=rec_c["site_pos"], phase_c_ball_pos=rec_c["ball_pos"],
        phase_c_ball_vel=rec_c["ball_vel"], phase_c_contact=rec_c["contact"],
        phase_c_site_vel=rec_c["site_vel"], phase_c_site_normal=rec_c["site_normal"],
        analysis=np.array(str(analysis), dtype=object),
        qpos_diff_pre=pre_diff,
    )
    print(f"[HitTest] Frame data saved to {out_file}")

    if not analysis.get("tunneling", False) and "first_contact_frame" in analysis:
        print("[HitTest] RESULT: contact + rebound OK "
              "(ball ejected with significant speed, no tunneling)")
        return 0
    print("[HitTest] RESULT: FAIL or tunneling suspected — "
          "retry with --sim_dt 0.0005 and/or --margin 0.002")
    return 1


if __name__ == "__main__":
    import tyro
    raise SystemExit(main(tyro.cli(HitTestArgs)))
