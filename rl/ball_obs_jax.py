"""jax 版球观测 / 弹道预测 / 发球采样（RL 阶段 1 · T2）。

与 numpy 参照实现逐公式对齐（对齐单测 tests/test_rl_ball.py，容差 1e-6）：
- utils/ball_predict.py: _accel / predict_ball_trajectory / predict_intercept /
  _integrate_state / fit_ball_state（refine_k=False 固定 k 分支 →
  fit_ball_state_fixed_k；refine_k=True 分支按 V1 结论不移植）
- utils/ball_obs.py: BallObserver 的 update / read_fit(refine_k=False) 路径；
  历史由 deque 改为定长环形缓冲 (H,3) + 步号 carry（jit/pytree 友好）
- utils/tennis_ball.py: sample_serve / design_serve_with_drag / apply_drag
  的外力公式（drag_force）

jit / vmap 设计：
- 全部纯函数；积分 = lax.scan；依赖 traced 数据的动态步数（观测窗时长、
  design 的 T+0.2 视界）用「静态 n_max + 掩码 i<n」扫描实现；
- 核心函数支持任意前导 batch 维（显式批量），亦可用 jax.vmap；
- jnp.linalg.lstsq 只接受 2D 输入：批量经 jax.vmap 逐样本求解（_lstsq）。
  jax CPU 实现为 QR 分解（numpy 的 LAPACK gelsd 是 SVD），本文件全部为
  小规模良态系统（6x6 正规方程 + 1e-9·I、(N,3) Vandermonde），数值稳定，
  实测与 numpy 对齐误差远小于 1e-6（单测有逐项打印）；
- 随机数显式注入：sample_serve_batch 接收 U[0,1) 均匀数组 u_*；
  BallObsBuffer.update 接收 rng key（update_with_draws 可注入确定性噪声）。
  jax 与 numpy 随机数流不同——对齐测试用「numpy 固定种子生成/录制的随机
  输入喂两版」，而不是比较同 seed 输出。

精度：本模块不再在 import 时全局开启 jax_enable_x64（与 MJX f32-only
管线冲突——同一进程 import 即污染）。默认全程 f32（RL 环境口径）；
需要与 numpy 位级对齐 / 1e-6 容差的场合（tests/test_rl_ball.py）显式
调用 enable_x64()（须先于任何 jnp 数组创建，进程级全局配置）。
"""

from __future__ import annotations

import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax import random as jrandom


def enable_x64() -> bool:
    """显式开启进程级 float64（对齐测试用），返回此前的开关状态。

    幂等。等价旧版「import 本模块即全局开启 x64」的行为，但改为调用方
    显式 opt-in。也可用环境变量 RL_BALL_X64=1 在 import 时自动开启
    （例如遗留 numpy 对齐脚本）。
    """
    prev = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    return prev


if os.environ.get("RL_BALL_X64", "").strip().lower() in ("1", "true", "yes"):
    enable_x64()

G = 9.81            # 重力（与 utils/ball_predict.py / utils/tennis_ball.py 一致）
BALL_MASS = 0.0577  # 球质量 kg（apply_drag 从 mj_model 读同一值；drag_force 默认）
DEFAULT_DT = 0.002  # 弹道积分步长（predict_ball_trajectory 默认）
SERVE_DT = 0.005    # design_serve_with_drag 积分步长

_GN_EPS = 1e-6      # GN 有限差分扰动（numpy 照抄）
_GN_TOL = 1e-6      # GN 收敛阈值 norm(delta) < 1e-6（numpy 照抄）
_LSTSQ_REG = 1e-9   # GN 正规方程正则项（numpy 照抄）

__all__ = [
    "BALL_MASS",
    "DEFAULT_DT",
    "G",
    "SERVE_DT",
    "BallObsBuffer",
    "BallObsState",
    "accel",
    "design_serve_with_drag",
    "drag_force",
    "duration_to_steps",
    "enable_x64",
    "fit_ball_state_fixed_k",
    "integrate_state",
    "integrate_trajectory",
    "predict_intercept",
    "sample_serve_batch",
    "step_euler",
]


# ---------------------------------------------------------------------------
# 基础积分器（对齐 utils/ball_predict.py）
# ---------------------------------------------------------------------------

def accel(vel, k=0.0):
    """重力 + 阻力加速度（对齐 _accel）。vel: (..., 3) 世界系速度。

    a = -G·e_z - k·|v|·v（k<=0 或 |v|<=1e-9 时无阻力项，与 numpy 的
    if k > 0 and speed > 1e-9 分支一致，用掩码实现以保持可微/jit）。
    """
    speed = jnp.linalg.norm(vel, axis=-1, keepdims=True)
    active = jnp.logical_and(k > 0.0, speed > 1e-9)
    coef = jnp.where(active, k * speed, 0.0)
    gravity = jnp.array([0.0, 0.0, -G], dtype=vel.dtype)
    return -coef * vel + gravity


def step_euler(pos, vel, k=0.0, dt=DEFAULT_DT):
    """半隐式欧拉一步：v += a·dt; p += v·dt（与 numpy 相同的更新顺序）。"""
    v_new = vel + accel(vel, k) * dt
    p_new = pos + v_new * dt
    return p_new, v_new


def integrate_trajectory(pos, vel, k=0.0, n_steps=1500, dt=DEFAULT_DT,
                         horizon=None):
    """前向积分弹道（对齐 predict_ball_trajectory，默认 1500 步 = 3.0s/0.002）。

    Args:
        pos/vel: (..., 3)；k: 阻力系数（标量，可 traced）。
        n_steps: 静态 int 步数（决定输出 shape）；或经 horizon=float 推导
            n_steps = int(horizon / dt)（对齐 numpy 的 n = int(h/dt)+1 点数）。
    Returns:
        (positions (n_steps+1, ..., 3), velocities (n_steps+1, ..., 3),
         times (n_steps+1,))；times[k] = k*dt（浮点乘积，与 numpy
        np.arange(n)*dt 逐位一致，非累加）。
    """
    if horizon is not None:
        n_steps = int(horizon / dt)
    if not isinstance(n_steps, int):
        raise TypeError("n_steps 必须是 Python int（静态量，决定输出 shape）")
    if n_steps < 0:
        raise ValueError("n_steps 必须 >= 0")

    def body(carry, _):
        p, v = carry
        p2, v2 = step_euler(p, v, k, dt)
        return (p2, v2), (p2, v2)

    _, (ps, vs) = lax.scan(body, (pos, vel), xs=None, length=n_steps)
    ps = jnp.moveaxis(ps, 0, -2)  # scan 时间轴置前 -> 移到 batch 维后（无 batch 时 no-op）
    vs = jnp.moveaxis(vs, 0, -2)
    all_p = jnp.concatenate([jnp.expand_dims(pos, -2), ps], axis=-2)
    all_v = jnp.concatenate([jnp.expand_dims(vel, -2), vs], axis=-2)
    ts = jnp.arange(n_steps + 1, dtype=all_p.dtype) * dt
    return all_p, all_v, ts


def duration_to_steps(duration, dt=DEFAULT_DT):
    """numpy 的 int(round(duration/dt))：round 为 half-even，jnp.rint 一致。"""
    return jnp.rint(jnp.asarray(duration) / dt).astype(jnp.int32)


def _integrate_traj_masked(pos, vel, k, n_steps, n_max, dt):
    """掩码扫描：n_max（静态）步中仅在 i < n_steps（可 traced）时积分。

    返回 (positions (..., n_max+1, 3), velocities)；k >= n_steps 的尾部冻结
    （值 = 第 n_steps 步后的状态），供规则网格插值在 [0, n_steps·dt] 内取值。
    调用方须保证 n_max >= 所有可能的 n_steps，否则静默截断。
    """
    def body(carry, i):
        p, v = carry
        p2, v2 = step_euler(p, v, k, dt)
        take = (i < n_steps)[..., None]  # 标量 -> (1,)；批量 (B,) -> (B,1)
        p = jnp.where(take, p2, p)
        v = jnp.where(take, v2, v)
        return (p, v), (p, v)

    _, (ps, vs) = lax.scan(body, (pos, vel), xs=jnp.arange(n_max), length=n_max)
    ps = jnp.moveaxis(ps, 0, -2)  # scan 时间轴置前 -> 移到 batch 维后（无 batch 时 no-op）
    vs = jnp.moveaxis(vs, 0, -2)
    all_p = jnp.concatenate([jnp.expand_dims(pos, -2), ps], axis=-2)
    all_v = jnp.concatenate([jnp.expand_dims(vel, -2), vs], axis=-2)
    return all_p, all_v


def integrate_state(pos, vel, k, n_steps, dt=DEFAULT_DT, n_steps_max=None):
    """半隐式欧拉积分 n_steps 步返回终态（对齐 _integrate_state）。

    n_steps 可为 traced 标量；此时必须传静态 n_steps_max（>= 所有可能步数，
    内部掩码扫描）。duration -> 步数换算用 duration_to_steps（half-even）。
    """
    if isinstance(n_steps, int):
        n_max = n_steps
    elif n_steps_max is not None:
        n_max = int(n_steps_max)
    else:
        raise ValueError("traced n_steps 需要静态 n_steps_max（掩码扫描上限）")
    n_steps = jnp.asarray(n_steps, dtype=jnp.int32)
    p_traj, v_traj = _integrate_traj_masked(pos, vel, k, n_steps, n_max, dt)
    return p_traj[..., -1, :], v_traj[..., -1, :]


def predict_intercept(pos, vel, z_plane, k=0.0, require_descending=True,
                      dt=DEFAULT_DT, n_steps=1500):
    """预测到达高度 z_plane 的（首个下降穿越）位置与时刻（对齐 predict_intercept）。

    沿 integrate_trajectory 找首个满足 z_prev >= z_plane > z_curr 的步，
    线性插值位置与时刻（分母 max(z_prev-z_curr, 1e-12) 与 numpy 一致）。
    numpy 版不可达时返回 (None, None)；jax 版返回 (pos, t, found)，
    found=False 时 pos/t 为 0（保持 jit 下形状静态、无 NaN 污染）。

    require_descending=True 时额外要求穿越步末速度 vz<0。半隐式欧拉下
    z_curr = z_prev + v_new·dt，故 z_curr < z_prev ⇔ v_new<0 —— 掩码与
    numpy 逻辑严格等价（numpy 源码两分支返回值本就相同，掩码显式化意图）。
    """
    p_traj, v_traj, ts = integrate_trajectory(pos, vel, k=k, n_steps=n_steps,
                                              dt=dt)
    z_prev = p_traj[..., :-1, 2]
    z_curr = p_traj[..., 1:, 2]
    cross = jnp.logical_and(z_prev >= z_plane, z_curr < z_plane)
    if require_descending:
        cross = jnp.logical_and(cross, v_traj[..., 1:, 2] < 0.0)
    found = jnp.any(cross, axis=-1)
    idx = jnp.argmax(cross, axis=-1)  # 首个 True（无 True 时为 0，由 found 掩掉）

    def _pair(a):  # (..., n, d) -> 取 idx 与 idx+1 两行
        lo = jnp.take_along_axis(a, idx[..., None, None], axis=-2)[..., 0, :]
        hi = jnp.take_along_axis(a, (idx + 1)[..., None, None], axis=-2)[..., 0, :]
        return lo, hi

    p_lo, p_hi = _pair(p_traj)
    z_lo = jnp.take_along_axis(z_prev, idx[..., None], axis=-1)[..., 0]
    z_hi = jnp.take_along_axis(z_curr, idx[..., None], axis=-1)[..., 0]
    frac = (z_lo - z_plane) / jnp.maximum(z_lo - z_hi, 1e-12)
    t_arrive = ts[idx] + frac * (ts[idx + 1] - ts[idx])
    p = p_lo + frac[..., None] * (p_hi - p_lo)
    p = jnp.where(found[..., None], p, jnp.zeros_like(p))
    t_arrive = jnp.where(found, t_arrive, jnp.zeros_like(t_arrive))
    return p, t_arrive, found


# ---------------------------------------------------------------------------
# 弹道拟合（对齐 fit_ball_state 的 refine_k=False 固定 k 分支）
# ---------------------------------------------------------------------------

def _interp_regular_cols(x, dt, fp):
    """规则网格线性插值（对齐 np.interp 的计算式，含端点 clamp）。

    x: (..., M) 查询时刻（>= 0）；fp: (..., n, 3) 网格值，节点 x_k = k*dt
    （与 numpy np.arange(n)*dt 逐位一致）。公式与 np.interp 相同：
    y_lo + (y_hi - y_lo) * (x - x_lo) / (x_hi - x_lo)。
    """
    n = fp.shape[-2]
    i = jnp.clip(jnp.floor(x / dt).astype(jnp.int32), 0, n - 2)
    x_lo = i * dt
    x_hi = (i + 1) * dt
    frac = (x - x_lo) / (x_hi - x_lo)
    y_lo = jnp.take_along_axis(fp, i[..., None], axis=-2)
    y_hi = jnp.take_along_axis(fp, (i + 1)[..., None], axis=-2)
    return y_lo + (y_hi - y_lo) * frac[..., None]


def _sample_positions(p0, v0, k, t_rel, n_steps_max, dt=DEFAULT_DT):
    """从 (p0, v0) 积分后在 t_rel 时刻插值采样（对齐 _sample_positions）。

    p0/v0: (..., 3)；t_rel: (..., N) 相对首帧时刻（升序）。numpy 视界
    horizon = t_rel[-1] + dt、步数 int(horizon/dt)；此处掩码扫描 n_steps_max
    步（n_steps = floor((t_rel[-1]+dt)/dt)，traced）+ 规则网格插值。
    """
    t_end = t_rel[..., -1]
    n_steps = jnp.floor((t_end + dt) / dt).astype(jnp.int32)
    p_traj, _ = _integrate_traj_masked(p0, v0, k, n_steps, int(n_steps_max), dt)
    return _interp_regular_cols(t_rel, dt, p_traj)


def _lstsq(A, b):
    """jnp.linalg.lstsq 的批量包装：2D 直接解，更高维经 vmap 逐样本。

    jnp.linalg.lstsq 只接受 2D（jax 0.8 限制；3D 直接调用报 "Array must be
    two-dimensional"），vmap 下可用（已实测）。本文件中系统均小规模良态
    （6x6 + 1e-9·I 正规方程、(N,3) Vandermonde），QR 解与 numpy 的 SVD 解
    对齐误差 ~1e-12 量级（单测逐项打印实测值）。
    """
    if A.ndim == 2:
        return jnp.linalg.lstsq(A, b, rcond=None)[0]
    return jax.vmap(_lstsq)(A, b)


def fit_ball_state_fixed_k(t_obs, p_obs, k=0.0, t_eval=None, mask=None,
                           dt=DEFAULT_DT, n_steps_max=1024):
    """固定 k 的 6 参数 Gauss-Newton 弹道拟合（对齐 fit_ball_state refine_k=False）。

    与 numpy 逐项一致的算法常数：二次 lstsq 初始化（k=0）→ 3 次 GN 迭代、
    有限差分 eps=1e-6、雅可比 (3N, 6)、正规方程 lstsq(J^T J + 1e-9 I,
    -J^T r)、收敛阈值 norm(delta) < 1e-6（提前停用 done 掩码等价实现：
    numpy 是「应用 delta 后检查」，此处已按同样顺序）。k<=0 时跳过 GN
    （gn_active 掩码，与 numpy 的 elif k>0 分支一致），k_est 恒等于入参 k。

    Args:
        t_obs: (..., N) 绝对时间（秒）；p_obs: (..., N, 3) 观测位置。
        k: 固定阻力系数（标量，可 traced；k_est == k）。
        t_eval: 求值时刻（标量，可 traced）；None = t_obs[..., -1]（最新观测，
            与 numpy read_fit 的默认语义一致）。
        mask: (..., N) 有效点掩码——False 的点等价于从拟合中剔除（行置零
            对正规方程无贡献），用于定长填充环形缓冲的窗口截取。
        n_steps_max: 静态 int，>= int((t_rel[-1]+dt)/dt) 的最大可能值
            （掩码扫描上限）。默认 1024（2.046s @ 0.002s/步）覆盖
            max_points=16 × ctrl_dt=0.033 的窗口。

    Returns:
        (pos (..., 3), vel (..., 3), k_est (...))，评估于 t_eval 时刻。
        numpy 版 n<4 的 ValueError / t_rel[-1]<=0 的早退分支在此由
        read_fit 的 ready 掩码 / where 分支承接（无异常路径）。
    """
    t_obs = jnp.asarray(t_obs)
    p_obs = jnp.asarray(p_obs)
    n = t_obs.shape[-1]
    if mask is None:
        mask = jnp.ones_like(t_obs, dtype=jnp.bool_)
    m3 = mask[..., None]
    t0 = t_obs[..., 0]
    t_rel = t_obs - t0[..., None]
    t_rel_last = t_rel[..., -1]

    # 1) 初始化：k=0 逐轴二次拟合（无效点行置零——正规方程视角等价于剔除）
    A = jnp.stack([jnp.ones_like(t_rel), t_rel, t_rel * t_rel], axis=-1) * m3
    Pm = p_obs * m3
    coef = _lstsq(A, Pm)  # (..., 3, 3)，行 0/1 = 各轴 p0/v0
    p0 = coef[..., 0, :]
    v0 = coef[..., 1, :]

    # numpy 早退分支：t_rel[-1] <= 0 时返回 (p_obs[-1], 0, k)
    early = t_rel_last <= 0.0
    p0 = jnp.where(early[..., None], p_obs[..., -1, :], p0)
    v0 = jnp.where(early[..., None], jnp.zeros_like(v0), v0)

    # 2) k>0：固定 k 的 6 参数 GN 细化（3 次迭代，常数照抄 numpy）
    gn_active = jnp.asarray(k) > 0.0  # k<=0 时 delta 恒 0（numpy 跳过 GN）
    mask_flat = jnp.repeat(mask, 3, axis=-1)  # (..., 3N)，与 (N,3).ravel() 同序

    def sample(pp, vv):
        return _sample_positions(pp, vv, k, t_rel, n_steps_max, dt)

    def gn_iter(carry, _):
        p0, v0, pred, done = carry
        resid = (pred - Pm).reshape(p_obs.shape[:-2] + (3 * n,)) * mask_flat
        base = jnp.concatenate([p0, v0], axis=-1)  # (..., 6)
        pert = base[..., None, :] + _GN_EPS * jnp.eye(6, dtype=base.dtype)
        pp = jnp.moveaxis(pert[..., :, :3], -2, 0)  # (6, ..., 3)
        vv = jnp.moveaxis(pert[..., :, 3:], -2, 0)  # (6, ..., 3)
        preds_p = jax.vmap(sample)(pp, vv)  # (6, ..., N, 3)
        diff = (preds_p - pred) / _GN_EPS
        # 列 j = 第 j 个扰动的残差差分展平（与 numpy J[:, j] = ...ravel()/eps 同序）
        J = jnp.reshape(jnp.moveaxis(diff, 0, -1),
                        p_obs.shape[:-2] + (3 * n, 6)) * mask_flat[..., None]
        # 正规方程（批量安全；与 numpy 的 J.T@J / J.T@r 等价）
        JTJ = jnp.einsum("...ji,...jk->...ik", J, J) \
            + _LSTSQ_REG * jnp.eye(6, dtype=J.dtype)
        delta = _lstsq(JTJ, -jnp.einsum("...ji,...j->...i", J, resid))  # (..., 6)
        nd = jnp.linalg.norm(delta, axis=-1)
        # numpy：应用 delta -> 重算 pred -> 检查 norm(delta)<1e-6 则 break
        apply = jnp.logical_and(jnp.logical_not(done), gn_active)
        upd = jnp.where(apply[..., None], delta, jnp.zeros_like(delta))
        p0n = p0 + upd[..., :3]
        v0n = v0 + upd[..., 3:]
        carry = (p0n, v0n, sample(p0n, v0n), jnp.logical_or(done, nd < _GN_TOL))
        return carry, None

    init_done = jnp.zeros_like(t0, dtype=jnp.bool_)
    (p0, v0, _, _), _ = lax.scan(gn_iter, (p0, v0, sample(p0, v0), init_done),
                                 xs=None, length=3)

    # 3) 评估到 t_eval（默认最新观测时刻；dur<=0 时返回 (p0, v0)）
    if t_eval is None:
        t_eval = t_obs[..., -1]
    dur = jnp.asarray(t_eval) - t0
    steps = jnp.rint(dur / dt).astype(jnp.int32)
    p_traj, v_traj = _integrate_traj_masked(p0, v0, k, steps, int(n_steps_max),
                                            dt)
    late = dur <= 0.0
    pos = jnp.where(late[..., None], p0, p_traj[..., -1, :])
    vel = jnp.where(late[..., None], v0, v_traj[..., -1, :])
    k_est = jnp.zeros_like(t0) + k
    return pos, vel, k_est


# ---------------------------------------------------------------------------
# 发球采样 / 阻力外力（对齐 utils/tennis_ball.py）
# ---------------------------------------------------------------------------

def drag_force(vel, k, m=BALL_MASS):
    """二次空气阻力外力 f = -k·m·v·|v|（对齐 apply_drag 的外力公式）。

    vel: (..., 3)；k: (...,) 阻力系数；m: 标量质量。k<=0 或 |v|<1e-9 时
    返回 0（numpy 端此时不施加外力，MuJoCo qfrc_applied 保持为 0）。
    """
    speed = jnp.linalg.norm(vel, axis=-1)
    active = jnp.logical_and(jnp.asarray(k) > 0.0, speed > 1e-9)
    f = -jnp.asarray(k)[..., None] * m * vel * speed[..., None]
    return jnp.where(active[..., None], f, jnp.zeros_like(f))


def design_serve_with_drag(target, T, k, serve_height, serve_back,
                           dt=SERVE_DT, n_steps_max=256):
    """带阻力的发射速度设计（对齐 utils/tennis_ball.py::design_serve_with_drag）。

    名义抛体解后 4 次迭代修正：predict(dt=0.005, horizon=T+0.2) 找 t=T
    位置（与 numpy 相同的 i = int(T/dt) 截断索引 + 线性插值），err =
    target - p_T，v += err/T。numpy 在 ‖err‖<1e-3 时于修正前 break——
    此处以 done 掩码等价实现（收敛轮起不再修正）。静态 4 次 lax.scan。

    Args:
        target: (..., 3)；T: (...,) 飞行时间；k: 阻力系数（与预测侧一致）。
        n_steps_max: 静态 int >= int((T_max + 0.2)/dt)（T_max = 调用方飞行
            时间上界；sample_serve_batch 会自动计算传入，默认 256 覆盖
            T <= 1.08s）。
    Returns:
        (p0 (..., 3), v (..., 3))。
    """
    target = jnp.asarray(target)
    T = jnp.asarray(T)
    p0 = target + jnp.array([serve_back, 0.0, serve_height], dtype=target.dtype)
    z_term = 0.5 * G * T * T  # python 左结合 (((0.5*G)*T)*T)，与 numpy 位级一致
    v = (target - p0 + z_term[..., None] * jnp.array([0.0, 0.0, 1.0],
                                                      dtype=target.dtype)
         ) / T[..., None]
    i = jnp.floor(T / dt).astype(jnp.int32)  # numpy int(T/0.005) 截断
    n_actual = jnp.floor((T + 0.2) / dt).astype(jnp.int32)

    def body(carry, _):
        v, done = carry
        p_traj, _ = _integrate_traj_masked(p0, v, k, n_actual, int(n_steps_max),
                                           dt)
        p_lo = jnp.take_along_axis(p_traj, (i - 1)[..., None, None],
                                   axis=-2)[..., 0, :]
        p_hi = jnp.take_along_axis(p_traj, i[..., None, None],
                                   axis=-2)[..., 0, :]
        frac = (T - (i - 1) * dt) / dt
        p_T = p_lo + frac[..., None] * (p_hi - p_lo)
        err = target - p_T
        conv = jnp.linalg.norm(err, axis=-1) < 1e-3
        apply = jnp.logical_not(jnp.logical_or(done, conv))
        v = v + jnp.where(apply[..., None], err / T[..., None],
                          jnp.zeros_like(err))
        return (v, jnp.logical_or(done, conv)), None

    init_done = jnp.zeros_like(T, dtype=jnp.bool_)
    (v, _), _ = lax.scan(body, (v, init_done), xs=None, length=4)
    return p0, v


def sample_serve_batch(p_swing, u_r, u_theta, u_z, u_T, feas_radius,
                       flight_lo, flight_hi, serve_height=1.0, serve_back=2.8,
                       design_k=0.0, dt=SERVE_DT, n_steps_max=None):
    """可行域发球采样（对齐 sample_serve），随机数显式注入。

    u_r/u_theta/u_z/u_T: (...,) 相互独立的 U[0,1) 样本。numpy 端对应的
    draw 顺序（录制对齐用）：rng.random() ×2 → uniform(-0.02, 0.02) →
    uniform(flight_lo, flight_hi)；均匀映射统一为 lo + (hi-lo)·u（与
    numpy Generator.uniform 的计算式一致，u_z/u_T 由录制值反解）。

    其余与 numpy 逐式一致：圆盘采样 r = R·sqrt(u)（盘上均匀）、z 偏移
    ±0.02、T 量化到 0.26+round((T-0.26)/0.02)·0.02（python round 是
    half-even，jnp.rint 同）并 clip 到 [lo-grid, hi+grid]；
    design_k>0 走带阻力迭代设计，否则抛体闭式解。

    Returns:
        (serve_pos (..., 3), serve_vel (..., 3), T_flight (...,),
         target (..., 3))。
    """
    u_r = jnp.asarray(u_r)
    u_theta = jnp.asarray(u_theta)
    u_z = jnp.asarray(u_z)
    u_T = jnp.asarray(u_T)
    r = feas_radius * jnp.sqrt(u_r)
    theta = u_theta * (2.0 * jnp.pi)
    z_off = -0.02 + 0.04 * u_z  # np uniform(lo,hi) = lo + (hi-lo)·u
    offset = jnp.stack([r * jnp.cos(theta), r * jnp.sin(theta), z_off], axis=-1)
    target = jnp.asarray(p_swing) + offset
    T = flight_lo + (flight_hi - flight_lo) * u_T
    grid = 0.02
    T = 0.26 + jnp.rint((T - 0.26) / grid) * grid
    T = jnp.clip(T, flight_lo - grid, flight_hi + grid)
    if design_k > 0:  # design_k 须为 Python float（静态分支，与 numpy 一致）
        if n_steps_max is None:
            # jit 下 flight_hi 须为静态/具体值；traced 时请显式传 n_steps_max
            n_steps_max = int((flight_hi + grid + 0.2) / dt) + 2
        p0, v = design_serve_with_drag(target, T, design_k, serve_height,
                                       serve_back, dt=dt, n_steps_max=n_steps_max)
    else:
        p0 = target + jnp.array([serve_back, 0.0, serve_height],
                                dtype=target.dtype)
        z_term = 0.5 * G * T * T  # python 左结合，与 numpy 位级一致
        v = (target - p0 + z_term[..., None] * jnp.array(
            [0.0, 0.0, 1.0], dtype=target.dtype)) / T[..., None]
    return p0, v, T, target


# ---------------------------------------------------------------------------
# 球观测环形缓冲（对齐 utils/ball_obs.py::BallObserver 的 update/read_fit）
# ---------------------------------------------------------------------------

class BallObsState(NamedTuple):
    """定长环形历史缓冲状态（jit/pytree 友好，替代 numpy 版 deque）。

    支持前导 batch 维 (...)：hist_pos (..., H, 3)、hist_step (..., H)
    int32（空槽 = -1）、last_valid (..., 3)、ptr (...,) int32 环形写指针。
    has_valid (...,) bool 对应 numpy 的 `_last_valid is not None`。
    """
    hist_pos: jnp.ndarray
    hist_step: jnp.ndarray
    last_valid: jnp.ndarray
    ptr: jnp.ndarray
    has_valid: jnp.ndarray


class BallObsBuffer:
    """球观测器：真值进、带噪观测出（对齐 BallObserver 的 update/read_fit）。

    与 numpy 版的差异（RL 批量化所需）：
    - 历史 = 定长环形数组 + 步号 carry（H = hist_len），非 deque；
    - 随机数显式注入：update(state, pos, step, key) 内部 split key 采样
      drop 判据与噪声；update_with_draws(...) 直接给定确定性噪声/判据
      （对齐测试把 numpy 录制的噪声喂进来用）。注意 numpy 版仅在
      `_last_valid is not None` 时才消耗 drop 判据，draw 流不同构，
      对齐测试只比确定性输入下的输出；
    - read_fit 返回 (pos, vel, k_est, ready)，ready=False 对应 numpy 版
      返回 None（可用观测 < max(4, min_points)）；k_est 恒等于入参 k
      （refine_k=False 固定 k 路径，RL V1 已验证）；
    - 状态评估于最新可用观测时刻（t_eval=None，消除 LSQ 中点滞后），
      compensate_delay 可传 t_eval = step*ctrl_dt 显式前推。
    """

    def __init__(self, hist_len=16, pos_sigma=0.005, dropout_p=0.05,
                 delay_frames=1, ctrl_dt=0.02):
        self.hist_len = int(hist_len)
        self.pos_sigma = float(pos_sigma)
        self.dropout_p = float(dropout_p)
        self.delay_frames = int(delay_frames)
        self.ctrl_dt = float(ctrl_dt)
        # read_fit 内 fit_ball_state_fixed_k 的掩码扫描上限：观测窗最大时长
        # = (hist_len-1)*ctrl_dt + dt（步长 0.002），+2 步余量
        self.n_fit_steps_max = int(
            ((self.hist_len - 1) * self.ctrl_dt + 0.01) / 0.002) + 2

    def make_init(self, batch_shape=()):
        """全空缓冲（hist_step 填 -1 哨兵 = 空槽）。"""
        H = self.hist_len
        return BallObsState(
            hist_pos=jnp.zeros(tuple(batch_shape) + (H, 3)),
            hist_step=jnp.full(tuple(batch_shape) + (H,), -1,
                               dtype=jnp.int32),
            last_valid=jnp.zeros(tuple(batch_shape) + (3,)),
            ptr=jnp.zeros(tuple(batch_shape), dtype=jnp.int32),
            has_valid=jnp.zeros(tuple(batch_shape), dtype=jnp.bool_),
        )

    def update_with_draws(self, state, true_pos, step, noise, drop_u):
        """确定性注入版 update：noise (..., 3) 已含幅度，drop_u (...,) 为
        U[0,1) 判据；drop ⇔ has_valid & (drop_u < dropout_p)，drop 时沿用
        last_valid（与 numpy 帧级 dropout 语义一致）。
        """
        drop = jnp.logical_and(state.has_valid, drop_u < self.dropout_p)
        cand = jnp.asarray(true_pos) + noise
        obs = jnp.where(drop[..., None], state.last_valid, cand)
        last_valid = jnp.where(drop[..., None], state.last_valid, cand)
        has_valid = jnp.ones_like(state.has_valid)
        idx = jnp.arange(self.hist_len)
        wmask = idx == state.ptr[..., None]  # (..., H) 环形写槽
        hist_pos = jnp.where(wmask[..., None], obs[..., None, :], state.hist_pos)
        hist_step = jnp.where(
            wmask, jnp.broadcast_to(jnp.asarray(step, dtype=jnp.int32),
                                    state.hist_step.shape), state.hist_step)
        ptr = (state.ptr + 1) % self.hist_len
        return BallObsState(hist_pos, hist_step, last_valid, ptr, has_valid)

    def update(self, state, true_pos, step, key):
        """采样版 update：key -> (drop_u, noise)。与 numpy 版同分布
        （噪声 lo+(hi-lo)·u 映射式相同），随机数流不同（对齐测试用
        update_with_draws）。"""
        key_drop, key_noise = jrandom.split(key)
        drop_u = jrandom.uniform(key_drop, state.has_valid.shape)
        u = jrandom.uniform(key_noise, jnp.asarray(true_pos).shape)
        noise = -self.pos_sigma + (2.0 * self.pos_sigma) * u
        return self.update_with_draws(state, true_pos, step, noise, drop_u)

    def read_fit(self, state, step, k=0.0, min_points=8, max_points=16,
                 t_eval=None):
        """模型约束读取（对齐 read_fit 的 refine_k=False 路径）。

        cutoff = step - delay_frames，取步号 <= cutoff 的最近 max_points 个
        观测（环形缓冲按步号排序后取尾部；不足处复制最旧有效点并经 mask
        剔除，等价于 numpy 的 avail[-max_points:] 切片），对窗口做
        fit_ball_state_fixed_k（固定 k），评估于最新可用观测时刻。

        Returns:
            (pos (..., 3), vel (..., 3), k_est (...,), ready (...,) bool)；
            ready=False（可用观测 < max(4, min_points)）时 pos/vel 为 0，
            对应 numpy 版返回 None。
        """
        H = self.hist_len
        M = int(max_points)
        cutoff = jnp.asarray(step, dtype=jnp.int32) - self.delay_frames
        valid = jnp.logical_and(state.hist_step >= 0,
                                state.hist_step <= cutoff[..., None])
        n_avail = jnp.sum(valid, axis=-1)
        ready = n_avail >= max(4, int(min_points))

        # 按「有效步号升序、无效槽（空槽 -1 / 未到期步）殿后」排序：
        # 有效块占据排序后前 n_avail 个位置，尾部窗口 gather 才能直接索引
        order = jnp.argsort(
            jnp.where(valid, state.hist_step, jnp.iinfo(jnp.int32).max),
            axis=-1, stable=True)
        sorted_step = jnp.take_along_axis(state.hist_step, order, axis=-1)
        sorted_pos = jnp.take_along_axis(state.hist_pos, order[..., None],
                                         axis=-2)
        j = jnp.arange(M)
        # 窗口 = 排序后第 [n_avail-M, n_avail) 个槽；不足 M 时前段复制最旧
        # 有效点（由 mask 剔除），保持静态 shape
        gather = jnp.clip(n_avail[..., None] - M + j, 0,
                          jnp.maximum(n_avail - 1, 0)[..., None])
        sel_step = jnp.take_along_axis(sorted_step, gather, axis=-1)  # (..., M)
        sel_pos = jnp.take_along_axis(sorted_pos, gather[..., None],
                                      axis=-2)  # (..., M, 3)
        mask_sel = j >= (M - n_avail[..., None])  # (..., M)

        t_obs = sel_step.astype(sel_pos.dtype) * self.ctrl_dt
        t_eval_arr = t_obs[..., -1] if t_eval is None else t_eval
        pos, vel, k_est = fit_ball_state_fixed_k(
            t_obs, sel_pos, k=k, t_eval=t_eval_arr, mask=mask_sel,
            n_steps_max=self.n_fit_steps_max)
        pos = jnp.where(ready[..., None], pos, jnp.zeros_like(pos))
        vel = jnp.where(ready[..., None], vel, jnp.zeros_like(vel))
        return pos, vel, k_est, ready
