"""冻结跟踪策略（pns_wo_priv264）ONNX -> flax 精确移植 + 观测/动作管线 jax 化。

包含三部分（均为推理用纯函数，不训练）：
1. FrozenPolicy：flax linen 模块，按 scripts/export_onnx_weights.py 导出的
   npz metadata（层宽/激活/权重）重建，ONNX 实际结构为
   136 -> 2048 -> 1024 -> 512 -> 29，激活 SiLU(x)=x*sigmoid(x) x3，
   输出层无激活（与文档宣称的 1024x3 + Sigmoid 不符，以 graph 为准）。
2. obs_jax：tracking/infer_utils.py 中 get_nn_state / update_coord_cmd /
   nn2motor_action 的 jax 纯函数版（支持批量，输入输出 f32）。
3. __main__ 自检：固定种子随机批（真实量级）flax vs onnxruntime 对齐。

用法：
    uv run python scripts/export_onnx_weights.py   # 先导出 npz（需 --with onnx）
    uv run python rl/frozen_policy.py              # 自检
"""

import json
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from tracking import constants as consts

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = REPO_ROOT / "storage/ckpts/pns_wo_priv264.onnx"
DEFAULT_NPZ = REPO_ROOT / "storage/ckpts/rl/pns_wo_priv264_flax.npz"

# 观测/动作维度（定义在 tracking/infer_utils.py，此处本地声明避免重依赖拉入 viewer）
NUM_STATE = 136  # 3+3+29+29+29+29+1+3+6+2+2
NUM_ACTION = 29  # len(ACTION_JOINT_NAMES)

_ACTIVATIONS = {
    "linear": lambda x: x,
    "silu": nn.silu,
    "sigmoid": jax.nn.sigmoid,
    "tanh": jnp.tanh,
    "relu": nn.relu,
}


# ---------------------------------------------------------------------------
# FrozenPolicy：纯 Dense/激活重建
# ---------------------------------------------------------------------------

class FrozenPolicy(nn.Module):
    """冻结 MLP 策略。features/activations 逐层对应，权重由 load_policy 从 npz 注入。"""

    features: tuple[int, ...]
    activations: tuple[str, ...]

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        x = obs
        for i, (feat, act) in enumerate(zip(self.features, self.activations)):
            x = nn.Dense(feat, name=f"dense_{i}")(x)
            x = _ACTIVATIONS[act](x)
        return x


def load_metadata(npz_path: str | Path = DEFAULT_NPZ) -> dict:
    """读取 npz 内的层结构 metadata json。"""
    with np.load(Path(npz_path)) as data:
        return json.loads(data["metadata"].item())


def load_policy(npz_path: str | Path = DEFAULT_NPZ) -> FrozenPolicy:
    """从 export_onnx_weights.py 导出的 npz 构建（已 bind 参数的）冻结策略。

    Returns:
        FrozenPolicy：已绑定参数，可直接 policy(obs) 调用（可 jax.jit）。
    """
    with np.load(Path(npz_path)) as data:
        meta = json.loads(data["metadata"].item())
        params = {"params": {}}
        for i, layer in enumerate(meta["layers"]):
            weight = data[layer["weight_key"]]  # Gemm transB=1: (out, in)
            kernel = weight.T if layer.get("trans_b", True) else weight
            params["params"][f"dense_{i}"] = {
                "kernel": np.ascontiguousarray(kernel, dtype=np.float32),
                "bias": np.asarray(data[layer["bias_key"]], dtype=np.float32),
            }
    model = FrozenPolicy(
        features=tuple(layer["out_features"] for layer in meta["layers"]),
        activations=tuple(layer["activation"] for layer in meta["layers"]),
    )
    return model.bind(jax.tree_util.tree_map(np.asarray, params))


def load_extras(npz_path: str | Path = DEFAULT_NPZ) -> dict[str, np.ndarray]:
    """加载主链之外的附加参数（如 std_param）。"""
    with np.load(Path(npz_path)) as data:
        meta = json.loads(data["metadata"].item())
        return {e["initializer"]: data[e["initializer"]] for e in meta.get("extras", [])}


# 本机(jax 0.8 CPU, Windows 沙箱)实测：batch>=16 的 f32 GEMM 内核执行会在 XLA CPU
# 并行路径上偶发死锁（编译 0.1s 即完成、执行永不返回；batch<=4 走串行路径 500+
# 次执行零失败）。因此批量前向按 4 行分块：语义不变，仅规避该执行路径。
FORWARD_CHUNK_ROWS = 4


def batch_forward(policy: FrozenPolicy, obs, chunk_rows: int = FORWARD_CHUNK_ROWS) -> np.ndarray:
    """冻结策略的批量前向（CPU 上按小行块执行，规避并行 GEMM 死锁路径）。

    Args:
        policy: load_policy() 返回的已绑定模块。
        obs: (N, 136) numpy/jnp 数组。
        chunk_rows: 每块行数，默认 4（串行内核路径）。

    Returns:
        (N, 29) numpy f32。
    """
    obs = np.asarray(obs, dtype=np.float32)
    chunk_fn = jax.jit(lambda o: policy(o))
    outs = []
    for i in range(0, obs.shape[0], chunk_rows):
        y = chunk_fn(jnp.asarray(obs[i:i + chunk_rows]))
        outs.append(np.asarray(y))
    return np.concatenate(outs, axis=0)


# ---------------------------------------------------------------------------
# obs_jax：get_nn_state / update_coord_cmd / nn2motor_action 的 jax 纯函数版
# ---------------------------------------------------------------------------

def _f32(x) -> jax.Array:
    return jnp.asarray(x, dtype=jnp.float32)


def _default_indices(num_jnt: int):
    qpos_ids = jnp.arange(7, 7 + num_jnt)     # 关节 qpos 列（qpos_ids_g1）
    dof_ids = jnp.arange(6, 6 + num_jnt)      # 关节 qvel 行（dof_ids_g1）
    ctrl_id_obs = jnp.arange(num_jnt)         # 观测序 -> 关节序（同名同序 = 恒等）
    return qpos_ids, dof_ids, ctrl_id_obs


def get_nn_state(
    gyro: jax.Array,
    gvec: jax.Array,
    qpos: jax.Array,
    qvel: jax.Array,
    ref_qpos: jax.Array,
    ref_kpt2gv_pose: jax.Array,
    ref_kpt_cvel_in_gv: jax.Array,
    last_action: jax.Array,
    yaw_d: jax.Array,
    xy_d: jax.Array,
    nominal: jax.Array = None,
    qpos_ids: jax.Array = None,
    dof_ids: jax.Array = None,
    ctrl_id_obs: jax.Array = None,
) -> jax.Array:
    """136 维观测向量的 jax 版（对齐 G1TrackInferFn.get_nn_state）。

    分量顺序：gyro(3)/gvec(3)/joint_qpos-nominal(29)/joint_qvel(29)/
    last_action(29)/ref_qpos-nominal(29)/ref_root_height(1)/ref_gvec(3)/
    ref_root_cvel(6)/yaw_cmd cos,sin(2)/xy_cmd(2)。

    Args:
        gyro, gvec: (B, 3) 骨盆陀螺仪 / 重力向量（gvec = -R^T e_z）。
        qpos: (B, nq) 完整广义位置（含基座，nq >= 36）。
        qvel: (B, nv) 完整广义速度（nv >= 35）。
        ref_qpos: (B, 36) 参考轨迹 qpos（ref_next 帧）。
        ref_kpt2gv_pose: (B, 14, 4, 4) 参考关键点在重力坐标系下位姿。
        ref_kpt_cvel_in_gv: (B, 14, 6) 参考关键点在重力坐标系下速度。
        last_action: (B, 29) 上一帧原始 nn 输出（隐状态）。
        yaw_d, xy_d: (B,) / (B, 2) update_coord_cmd 输出的指令。
        nominal: (29,) 默认关节位置，默认 consts.DEFAULT_QPOS[7:]。
        qpos_ids/dof_ids/ctrl_id_obs: 索引映射，默认按同名同序（恒等）。

    Returns:
        (B, 136) f32 观测向量。
    """
    num_jnt = NUM_ACTION
    if nominal is None:
        nominal = jnp.asarray(consts.DEFAULT_QPOS[7:], dtype=jnp.float32)
    if qpos_ids is None or dof_ids is None or ctrl_id_obs is None:
        d_qpos, d_dof, d_ctrl = _default_indices(num_jnt)
        qpos_ids = d_qpos if qpos_ids is None else qpos_ids
        dof_ids = d_dof if dof_ids is None else dof_ids
        ctrl_id_obs = d_ctrl if ctrl_id_obs is None else ctrl_id_obs

    gyro, gvec = _f32(gyro), _f32(gvec)
    qpos, qvel = _f32(qpos), _f32(qvel)
    last_action = _f32(last_action)
    nom = _f32(nominal)  # (29,)

    joint_qpos = qpos[..., qpos_ids][..., ctrl_id_obs] - nom
    joint_qvel = qvel[..., dof_ids][..., ctrl_id_obs]

    ref_jnt = _f32(ref_qpos)[..., 7:][..., ctrl_id_obs] - nom
    root_pose = _f32(ref_kpt2gv_pose)[..., 0, :, :]        # (B, 4, 4)
    ref_height = root_pose[..., 2, 3:4]                    # (B, 1)
    ref_gvec = -jnp.swapaxes(root_pose[..., :3, :3], -1, -2)[..., 2]  # -R^T e_z
    ref_cvel = _f32(ref_kpt_cvel_in_gv)[..., 0, :]         # (B, 6)

    yaw_d = _f32(yaw_d)
    yaw_cmd = jnp.stack([jnp.cos(yaw_d), jnp.sin(yaw_d)], axis=-1)  # (B, 2)

    state = jnp.concatenate(
        [
            gyro,                # 3
            gvec,                # 3
            joint_qpos,          # 29
            joint_qvel,          # 29
            last_action,         # 29
            ref_jnt,             # 29
            ref_height,          # 1
            ref_gvec,            # 3
            ref_cvel,            # 6
            yaw_cmd,             # 2
            _f32(xy_d),          # 2
        ],
        axis=-1,
    )
    assert state.shape[-1] == NUM_STATE, state.shape
    return state


def warp_pi(ang: jax.Array) -> jax.Array:
    """角度包装到 [-pi, pi]（对齐 transforms_np.WarpPi）。"""
    return jnp.arctan2(jnp.sin(ang), jnp.cos(ang))


def quat2yaw(quat_wxyz: jax.Array) -> jax.Array:
    """四元数 (w,x,y,z) -> 偏航角（对齐 transforms_np.quat2yaw）。"""
    w, x, y, z = (quat_wxyz[..., i] for i in range(4))
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r10 = 2.0 * (x * y + w * z)
    return warp_pi(jnp.arctan2(r10, r00))


def _rot2d(yaw: jax.Array, v: jax.Array) -> jax.Array:
    """v (..., 2) 逆时针旋转 yaw（对齐 update_coord_cmd 的 R(yaw) @ v）。"""
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    return jnp.stack([c * v[..., 0] - s * v[..., 1],
                      s * v[..., 0] + c * v[..., 1]], axis=-1)


def update_coord_cmd(
    qpos: jax.Array,
    ref_curr_qpos: jax.Array,
    yaw_cmd: jax.Array = None,
    xy_cmd: jax.Array = None,
) -> tuple[jax.Array, jax.Array]:
    """坐标指令的 jax 版（对齐 G1TrackInferFn.update_coord_cmd，根位姿纯函数）。

    yaw_d = WarpPi(yaw_cmd + yaw_ref - yaw_curr)；
    xy_d 先在世界系加偏移再旋到机器人系（两次 2D 旋转）。

    Args:
        qpos: (B, nq) 当前完整 qpos（仅用根四元数与根 xy）。
        ref_curr_qpos: (B, 36) 参考当前帧 qpos。
        yaw_cmd, xy_cmd: 附加指令偏移，默认 0。

    Returns:
        (yaw_d (B,), xy_d (B, 2))。
    """
    qpos = _f32(qpos)
    ref = _f32(ref_curr_qpos)
    yaw_ref = quat2yaw(ref[..., 3:7])
    yaw_curr = quat2yaw(qpos[..., 3:7])
    if yaw_cmd is None:
        yaw_cmd = jnp.zeros_like(yaw_curr)
    yaw_cmd = _f32(yaw_cmd)
    yaw_d = warp_pi(yaw_cmd + yaw_ref - yaw_curr)

    if xy_cmd is None:
        xy_cmd = jnp.zeros_like(_f32(qpos[..., :2]))
    xy_cmd = _f32(xy_cmd)
    xy_target = _rot2d(yaw_cmd, xy_cmd + ref[..., :2])   # 世界系目标
    xy_d = _rot2d(-yaw_curr, xy_target - qpos[..., :2])  # 转到机器人系
    return yaw_d, xy_d


def nn2motor_action(
    nn_action: jax.Array,
    nominal: jax.Array = None,
    ctrl_id_act: jax.Array = None,
    action_scale: float = 0.25,
    act_scale: jax.Array = None,
) -> jax.Array:
    """nn 输出 -> 电机目标位置（对齐 G1TrackInferFn.nn2motor_action）。

    motor[..., ctrl_id_act] = nominal[..., ctrl_id_act]
        + nn_action * action_scale * act_scale[ctrl_id_act]（无 clip）。

    Args:
        nn_action: (B, 29) 网络原始输出。
        nominal: (29,) 默认关节位置。
        ctrl_id_act: 动作关节索引（默认恒等）。
        action_scale: env_config.action_scale（0.25）。
        act_scale: (29,) consts.ACTION_SCALE = TORQUE_LIMIT / KPs。

    Returns:
        (B, 29) f32 电机目标位置。
    """
    nn_action = _f32(nn_action)
    if nominal is None:
        nominal = jnp.asarray(consts.DEFAULT_QPOS[7:], dtype=jnp.float32)
    if act_scale is None:
        act_scale = jnp.asarray(consts.ACTION_SCALE, dtype=jnp.float32)
    if ctrl_id_act is None:
        ctrl_id_act = jnp.arange(NUM_ACTION)
    nom = jnp.broadcast_to(_f32(nominal), nn_action.shape)
    delta = nn_action * action_scale * _f32(act_scale)[ctrl_id_act]
    return nom.at[..., ctrl_id_act].set(nom[..., ctrl_id_act] + delta)


# ---------------------------------------------------------------------------
# 随机观测合成（真实量级）+ __main__ 自检
# 观测布局（136 维）：
#   gyro(0:3) gvec(3:6) joint_qpos-nom(6:35) joint_qvel(35:64)
#   last_action(64:93) ref_qpos-nom(93:122) ref_height(122:123)
#   ref_gvec(123:126) ref_cvel(126:132) yaw cos/sin(132:134) xy_cmd(134:136)
# ---------------------------------------------------------------------------

def sample_realistic_obs(key: jax.Array, n: int) -> jax.Array:
    """固定种子合成 n 个真实量级的 136 维观测。

    注意：纯合成 obs 会离开策略训练数据流形，网络输出可达 ±35；此时
    onnxruntime 自身 f32 GEMM 累加误差（vs f64 精确值）就有 ~2.3e-5，
    任何重实现都无法在输出处通过 1e-5 绝对对齐——需要严格对齐验证时
    请用 sample_obs_near_frames()（录制帧附近采样，输出保持真实量级）。
    """
    keys = jax.random.split(key, 11)
    u = lambda i, shape, lo, hi: jax.random.uniform(
        keys[i], shape, minval=lo, maxval=hi, dtype=jnp.float32)
    def unit(i, shape):
        v = jax.random.normal(keys[i], shape, dtype=jnp.float32)
        return v / jnp.linalg.norm(v, axis=-1, keepdims=True)

    blocks = [
        u(0, (n, 3), -3.0, 3.0),                       # gyro ±3 rad/s
        unit(1, (n, 3)),                               # gvec 单位向量
        u(2, (n, 29), -1.0, 1.0),                      # joint_qpos-nominal ±1
        u(3, (n, 29), -5.0, 5.0),                      # joint_qvel ±5 rad/s
        u(4, (n, 29), -1.0, 1.0),                      # last_action ∈ [-1,1]
        u(5, (n, 29), -1.0, 1.0),                      # ref_qpos-nominal ±1
        0.8 + 0.1 * jax.random.normal(keys[6], (n, 1), dtype=jnp.float32),  # 高度 ~0.8m
        unit(7, (n, 3)),                               # ref_gvec 单位向量
        u(8, (n, 6), -3.0, 3.0),                       # ref 根速度 ±3
        None,                                          # yaw cos/sin（下方填）
        u(10, (n, 2), -2.0, 2.0),                      # xy_cmd ±2 m
    ]
    ang = u(9, (n,), -jnp.pi, jnp.pi)
    blocks[9] = jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=-1)
    obs = jnp.concatenate(blocks, axis=-1)
    assert obs.shape == (n, NUM_STATE)
    return obs


# 每分量抖动 σ（136 维布局，用于在录制帧附近采样物理有效的随机观测）：
#   gyro(0:3) gvec(3:6) joint_qpos(6:35) joint_qvel(35:64) last_action(64:93)
#   ref_qpos(93:122) ref_height(122:123) ref_gvec(123:126) ref_cvel(126:132)
#   yaw(132:134, 重新采角度) xy(134:136)
_JITTER_SIGMA = np.array(
    [0.3] * 3 + [0.02] * 3 + [0.1] * 29 + [0.3] * 29 + [0.2] * 29
    + [0.1] * 29 + [0.02] + [0.02] * 3 + [0.2] * 6 + [0.0] * 2 + [0.1] * 2,
    dtype=np.float32)


def sample_obs_near_frames(centers: np.ndarray, n: int = 4096, seed: int = 0) -> np.ndarray:
    """以真实录制 obs 为中心加每分量抖动，合成 n 个随机观测（numpy，固定种子）。

    gvec/ref_gvec 重归一化保持单位向量，yaw 重采角度保持单位圆，
    保证采样既物理有效又不离开策略工作流形（输出量级与真实数据一致）。
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(centers), size=n, replace=True)
    obs = centers[idx] + rng.normal(0.0, 1.0, (n, NUM_STATE)).astype(np.float32) * _JITTER_SIGMA
    for sl in (slice(3, 6), slice(123, 126)):
        obs[:, sl] /= np.linalg.norm(obs[:, sl], axis=-1, keepdims=True)
    ang = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    obs[:, 132:134] = np.stack([np.cos(ang), np.sin(ang)], axis=-1)
    return obs.astype(np.float32)


if __name__ == "__main__":
    import onnxruntime as rt

    meta = load_metadata()
    print(f"npz 层结构: {meta['input']['dim']} -> "
          + " -> ".join(str(l["out_features"]) for l in meta["layers"])
          + f" | 激活 {[l['activation'] for l in meta['layers']]}")

    policy = load_policy()
    rec_path = REPO_ROOT / "storage/data/rl/recorded_frames.npz"
    if rec_path.exists():
        with np.load(rec_path) as rec:
            obs = sample_obs_near_frames(rec["obs"], n=4096, seed=0)
        tol = 1e-5  # 流形内输出量级真实，可过严格阈值
    else:
        # 纯合成 obs 离开数据流形（|out| 可达 ±35，ORT 自身 f32 误差 ~2.3e-5），
        # 放宽为 1e-4；严格 1e-5 见 tests/test_rl_policy.py
        obs = np.asarray(sample_realistic_obs(jax.random.PRNGKey(0), 4096))
        tol = 1e-4
    sess = rt.InferenceSession(str(DEFAULT_ONNX), providers=["CPUExecutionProvider"])
    ref = sess.run([meta["output"]["name"]], {meta["input"]["name"]: obs})[0]
    out = batch_forward(policy, obs)
    assert out.shape == ref.shape == (4096, NUM_ACTION)

    max_abs = float(np.abs(out - ref).max())
    p99 = float(np.percentile(np.abs(out - ref), 99))
    print(f"随机批 4096 对齐: max|Δ|={max_abs:.3e}  p99|Δ|={p99:.3e} (tol {tol:.0e})")
    assert max_abs < tol, f"flax vs onnxruntime 偏差过大: {max_abs}"

    # obs_jax 冒烟：单批执行 + 形状检查
    qpos = jnp.tile(jnp.asarray(consts.DEFAULT_QPOS), (4, 1))
    qvel = jnp.zeros((4, consts.DEFAULT_QPOS.shape[0] - 1))
    yaw_d, xy_d = update_coord_cmd(qpos, qpos)
    state = get_nn_state(
        jnp.zeros((4, 3)), jnp.tile(jnp.array([0.0, 0.0, -1.0]), (4, 1)),
        qpos, qvel, qpos, jnp.broadcast_to(jnp.eye(4), (4, 14, 4, 4)),
        jnp.zeros((4, 14, 6)), jnp.zeros((4, 29)), yaw_d, xy_d)
    motor = nn2motor_action(policy(state))
    assert state.shape == (4, 136) and motor.shape == (4, 29)
    print(f"obs_jax 冒烟: obs{state.shape} motor{motor.shape} OK")
    print("frozen_policy 自检通过")
