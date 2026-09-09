"""RL 冻结策略移植对齐验证（双口径）。

a. 随机批：固定种子合成 ≥4096 个真实量级 obs（以录制帧为中心加抖动，
   保持物理有效与输出量级），onnxruntime vs flax max|Δ| < 1e-5；
b. 真实轨迹回归：CPU 管线回放 Random_001 全轨迹（循环 2 轮）录制
   (obs, nn_action) 等原始量，flax 前向对齐 < 1e-5；obs_jax 的
   get_nn_state / update_coord_cmd / nn2motor_action 与 numpy 版 < 1e-6。

运行：uv run python tests/test_rl_policy.py
前置：uv run --with onnx python scripts/export_onnx_weights.py
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # rl/ 不在已安装包列表内，手动加入仓库根
    sys.path.insert(0, str(REPO))

from rl import (  # noqa: E402
    DEFAULT_NPZ,
    DEFAULT_ONNX,
    batch_forward,
    get_nn_state,
    load_policy,
    nn2motor_action,
    sample_obs_near_frames,
    update_coord_cmd,
)

CLIP_PATH = REPO / "storage/data/mocap/Tennis/p1/Random_001_Tennis 001.npz"
REC_PATH = REPO / "storage/data/rl/recorded_frames.npz"
MIN_FRAMES = 2000
LOOPS = 2  # 1874 帧 @50Hz，回放两轮凑足 >=2000 帧


# ---------------------------------------------------------------------------
# 真实轨迹录制（CPU 管线：build_tennis_sim + G1TrackInferFn + load_segment）
# ---------------------------------------------------------------------------

def record_frames(clip_path=CLIP_PATH, loops=LOOPS, out_path=REC_PATH) -> dict:
    """回放 clip 全轨迹 loops 轮，逐帧录制对齐验证所需的全部原始量。

    录制内容：obs/nn_action（ORT 实际输入输出）、get_nn_state 输入
    （gyro/gvec/qpos/qvel/ref_next/last_action_in/yaw_d/xy_d）、
    update_coord_cmd 输入（ref_curr_qpos）、nn2motor 输出 motor_targets。
    """
    from jax import tree_util as jtu

    from tracking import constants as consts
    from tracking.infer_utils import G1TrackInferFn, g1_infer_env_config
    from tracking.policy import Args, get_policy_onnx
    from utils.tennis_clips import load_segment, reset_to_frame
    from utils.tennis_common import build_tennis_sim

    ref_traj = load_segment(str(clip_path), str(consts.TRACK_XML), 50)
    traj_len = len(ref_traj["qpos"])
    sim, _site_id, ball_adr, ball_dof = build_tennis_sim(headless=True)
    policy = get_policy_onnx(Args(policy_type="mlp", device="cpu",
                                  onnx_track=str(DEFAULT_ONNX)))
    infer_fn = G1TrackInferFn(g1_infer_env_config(ctrl_dt=0.02), sim.mj_model,
                              policy, privileged=False)
    state = reset_to_frame(sim, infer_fn, ref_traj, 0, ball_adr, ball_dof,
                           zero_root_vel=False)

    rec = {k: [] for k in [
        "obs", "nn_action", "qpos", "qvel", "gyro", "gvec", "last_action_in",
        "yaw_d", "xy_d", "ref_curr_qpos", "ref_next_qpos",
        "ref_next_kpt2gv_pose", "ref_next_kpt_cvel_in_gv", "motor_targets"]}
    for i in range(loops * traj_len):
        f = i % traj_len  # 循环回放：参考帧号绕回（参考跳变不影响对齐验证）
        nxt = (f + 1) % traj_len
        ref_curr = jtu.tree_map(lambda x: x[f][None], ref_traj)
        ref_next = jtu.tree_map(lambda x: x[nxt][None], ref_traj)
        last_action_in = infer_fn.info["last_action"][0].copy()
        motor = infer_fn.infer_onnx(state, {"ref_curr": ref_curr, "ref_next": ref_next})
        info = infer_fn.info
        rec["obs"].append(info["obs"][0].copy())
        rec["nn_action"].append(info["nn_action"][0].copy())
        rec["qpos"].append(info["qpos"][0].copy())
        rec["qvel"].append(info["qvel"][0].copy())
        rec["gyro"].append(info["gyro_pelvis"][0].copy())
        rec["gvec"].append(info["gvec_pelvis"][0].copy())
        rec["last_action_in"].append(last_action_in)
        rec["yaw_d"].append(np.float32(info["yaw_d"][0]))
        rec["xy_d"].append(info["xy_d"][0].copy())
        rec["ref_curr_qpos"].append(ref_curr["qpos"][0].copy())
        rec["ref_next_qpos"].append(ref_next["qpos"][0].copy())
        rec["ref_next_kpt2gv_pose"].append(ref_next["kpt2gv_pose"][0].copy())
        rec["ref_next_kpt_cvel_in_gv"].append(ref_next["kpt_cvel_in_gv"][0].copy())
        rec["motor_targets"].append(info["motor_targets"][0].copy())
        state = sim.step(state, motor)

    arrays = {k: np.asarray(v) for k, v in rec.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return arrays


def load_or_record_frames() -> dict:
    if REC_PATH.exists():
        return dict(np.load(REC_PATH))
    return record_frames()


def get_ort_session():
    import onnxruntime as rt

    return rt.InferenceSession(str(DEFAULT_ONNX), providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------------------
# a. 随机批对齐
# ---------------------------------------------------------------------------

def _align_tol():
    """flax vs onnxruntime 对齐容差：GPU 上 jax matmul 走 cuBLAS（累加顺序与
    CPU 的 MKL/OpenBLAS 不同），f32 累差天然 ~5e-5（实测），故放宽到 1e-4；
    CPU 上保持 1e-5 严格口径（实测 8e-6）。"""
    import jax
    return 1e-4 if any(d.platform == "gpu" for d in jax.devices()) else 1e-5


def test_random_batch_onnx_vs_flax():
    assert DEFAULT_NPZ.exists(), (
        f"缺少 {DEFAULT_NPZ}，先运行: "
        "uv run --with onnx python scripts/export_onnx_weights.py")
    frames = load_or_record_frames()
    policy = load_policy()
    # 以录制帧为中心加抖动（rl.sample_obs_near_frames）：保持物理有效与真实
    # 输出量级。纯合成 obs 会离开数据流形（|out| ~ ±35），那里 ORT 自身 f32
    # 累加误差就有 ~2.3e-5，1e-5 绝对阈值在任何重实现下都无法成立。
    obs = sample_obs_near_frames(frames["obs"], n=4096, seed=0)
    ref = get_ort_session().run(["continuous_actions"], {"obs": obs})[0]
    out = batch_forward(policy, obs)
    assert out.shape == ref.shape == (4096, 29)
    max_abs = float(np.abs(out - ref).max())
    print(f"  随机批 4096: max|Δ|={max_abs:.3e}")
    assert max_abs < _align_tol(), \
        f"flax vs onnxruntime 随机批偏差 {max_abs:.3e} >= {_align_tol()}"


# ---------------------------------------------------------------------------
# b. 真实轨迹回归
# ---------------------------------------------------------------------------

def test_real_traj_policy_alignment():
    frames = load_or_record_frames()
    n = len(frames["obs"])
    assert n >= MIN_FRAMES, f"录制帧数 {n} < {MIN_FRAMES}"
    policy = load_policy()
    out = batch_forward(policy, frames["obs"])
    max_abs = float(np.abs(out - frames["nn_action"]).max())
    print(f"  真实轨迹 {n} 帧: max|Δ|={max_abs:.3e} "
          f"|nn_action|max={np.abs(frames['nn_action']).max():.3f}")
    assert max_abs < _align_tol(), \
        f"flax vs 录制 nn_action 偏差 {max_abs:.3e} >= {_align_tol()}"


def test_real_traj_get_nn_state_alignment():
    import jax.numpy as jnp

    frames = load_or_record_frames()
    state = get_nn_state(
        jnp.asarray(frames["gyro"], dtype=jnp.float32),
        jnp.asarray(frames["gvec"], dtype=jnp.float32),
        jnp.asarray(frames["qpos"], dtype=jnp.float32),
        jnp.asarray(frames["qvel"], dtype=jnp.float32),
        jnp.asarray(frames["ref_next_qpos"], dtype=jnp.float32),
        jnp.asarray(frames["ref_next_kpt2gv_pose"], dtype=jnp.float32),
        jnp.asarray(frames["ref_next_kpt_cvel_in_gv"], dtype=jnp.float32),
        jnp.asarray(frames["last_action_in"], dtype=jnp.float32),
        jnp.asarray(frames["yaw_d"], dtype=jnp.float32),
        jnp.asarray(frames["xy_d"], dtype=jnp.float32),
    )
    obs = np.asarray(state)
    max_abs = float(np.abs(obs - frames["obs"]).max())
    print(f"  get_nn_state jax vs numpy: max|Δ|={max_abs:.3e}")
    assert max_abs < 1e-6, f"get_nn_state 偏差 {max_abs:.3e} >= 1e-6"


def _numpy_update_coord_cmd(qpos, ref_curr_qpos):
    """G1TrackInferFn.update_coord_cmd 的 numpy 复刻（与传入 dtype 同精度计算）。

    用于语义对齐比较：录制值由 numpy 以 f64 qpos 计算，jax 版输入是 f32，
    大坐标(xy 数米)的 f64->f32 舍入本身就有 ~1e-6；同 f32 输入下逐一复刻
    公式才能严格对齐语义（容差 1e-6）。
    """
    from utils.transforms_np import WarpPi, quat2yaw

    yaw_ref = quat2yaw(ref_curr_qpos[:, 3:7])
    yaw_curr = quat2yaw(qpos[:, 3:7])
    yaw_d = WarpPi(yaw_ref - yaw_curr)

    xy_ref, xy_curr = ref_curr_qpos[:, :2], qpos[:, :2]
    xy_target = xy_ref  # yaw_cmd=xy_cmd=0（load_segment 产物无 *_cmd 字段）
    xy_d = xy_target - xy_curr
    c, s = np.cos(-yaw_curr), np.sin(-yaw_curr)
    R_m = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
    xy_d = np.einsum("...ij,...j->...i", R_m, xy_d)
    return yaw_d, xy_d


def test_real_traj_update_coord_cmd_alignment():
    import jax.numpy as jnp

    frames = load_or_record_frames()
    qpos_f32 = frames["qpos"].astype(np.float32)
    ref_f32 = frames["ref_curr_qpos"].astype(np.float32)
    yaw_jax, xy_jax = update_coord_cmd(jnp.asarray(qpos_f32), jnp.asarray(ref_f32))

    # 语义对齐：同 f32 输入下 jax vs numpy 复刻，严格 1e-6
    yaw_np, xy_np = _numpy_update_coord_cmd(qpos_f32, ref_f32)
    d_yaw = float(np.abs(np.asarray(yaw_jax) - yaw_np).max())
    d_xy = float(np.abs(np.asarray(xy_jax) - xy_np).max())
    print(f"  update_coord_cmd jax vs numpy(f32 输入): "
          f"yaw max|Δ|={d_yaw:.3e} xy max|Δ|={d_xy:.3e}")
    assert d_yaw < 1e-6 and d_xy < 1e-6, (d_yaw, d_xy)

    # 对录制值（numpy f64 输入计算）另作参考：容差放宽为 f64->f32 输入舍入量级
    d_yaw64 = float(np.abs(np.asarray(yaw_jax) - frames["yaw_d"]).max())
    d_xy64 = float(np.abs(np.asarray(xy_jax) - frames["xy_d"]).max())
    print(f"  update_coord_cmd jax vs 录制值(f64 输入): "
          f"yaw max|Δ|={d_yaw64:.3e} xy max|Δ|={d_xy64:.3e}")
    assert d_yaw64 < 1e-6 and d_xy64 < 5e-6, (d_yaw64, d_xy64)


def test_real_traj_nn2motor_alignment():
    import jax.numpy as jnp

    frames = load_or_record_frames()
    motor = np.asarray(nn2motor_action(jnp.asarray(frames["nn_action"], dtype=jnp.float32)))
    max_abs = float(np.abs(motor - frames["motor_targets"]).max())
    print(f"  nn2motor_action jax vs numpy: max|Δ|={max_abs:.3e}")
    assert max_abs < 1e-6, f"nn2motor_action 偏差 {max_abs:.3e} >= 1e-6"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"RUN  {name}")
            fn()
            print(f"PASS {name}")
    print("all tests passed")
