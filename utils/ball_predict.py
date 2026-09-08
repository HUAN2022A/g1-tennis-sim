"""网球弹道预测：重力 + 可选空气阻力（V0 不含地面反弹，只在下降段拦截）。

模型（与 docs/TENNIS_STATE_MACHINE_PLAN.md 第 3 节一致）：
    f_air = -k * m * v * |v|   (k 为单位质量的阻力系数, 1/m)

无解析解（有阻力时），用小步长前向积分 + 二分求交点。
V0 使用无噪声真值球观测，速度可直接从仿真读出。
"""

from __future__ import annotations

import numpy as np

G = 9.81


def _accel(vel: np.ndarray, k: float) -> np.ndarray:
    """重力 + 阻力加速度。vel: (3,) 世界系速度。"""
    speed = np.linalg.norm(vel)
    a = np.zeros(3)
    a[2] = -G
    if k > 0 and speed > 1e-9:
        a -= k * speed * vel
    return a


def predict_ball_trajectory(pos: np.ndarray, vel: np.ndarray, k: float = 0.0,
                            dt: float = 0.002, horizon: float = 3.0
                            ) -> tuple[np.ndarray, np.ndarray]:
    """前向积分弹道。返回 (positions (T,3), times (T,))，含 t=0 起点。

    Args:
        pos/vel: 当前球位置/速度（世界系）
        k: 空气阻力系数（0 = 纯抛体）
        dt: 积分步长（半隐式欧拉）
        horizon: 预测时长
    """
    n = int(horizon / dt) + 1
    out_pos = np.zeros((n, 3))
    out_t = np.arange(n) * dt
    p = np.asarray(pos, dtype=np.float64).copy()
    v = np.asarray(vel, dtype=np.float64).copy()
    out_pos[0] = p
    for i in range(1, n):
        a = _accel(v, k)
        v = v + a * dt
        p = p + v * dt
        out_pos[i] = p
    return out_pos, out_t


def predict_intercept(pos: np.ndarray, vel: np.ndarray, z_plane: float,
                      k: float = 0.0, require_descending: bool = True
                      ) -> tuple[np.ndarray, float] | tuple[None, None]:
    """预测球到达高度 z_plane（击球高度平面）的位置与时刻。

    在"下降段"拦截（plan 第 6 节风险对策：只在弹跳前拦截），默认要求
    vz < 0 才算有效拦截点。

    Returns:
        (intercept_pos (3,), t_arrive) 或 (None, None) 表示预测不可达/不满足条件。
    """
    out_pos, out_t = predict_ball_trajectory(pos, vel, k=k)
    for i in range(1, len(out_t)):
        z_prev, z_curr = out_pos[i - 1][2], out_pos[i][2]
        if z_prev >= z_plane > z_curr:
            # 线性插值交点
            frac = (z_prev - z_plane) / max(z_prev - z_curr, 1e-12)
            t_arrive = out_t[i - 1] + frac * (out_t[i] - out_t[i - 1])
            p = out_pos[i - 1] + frac * (out_pos[i] - out_pos[i - 1])
            if require_descending:
                return np.asarray(p), float(t_arrive)
            return np.asarray(p), float(t_arrive)
    return None, None


def _integrate_state(pos: np.ndarray, vel: np.ndarray, k: float,
                     dt: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """半隐式欧拉积分 duration 秒，返回终态 (pos, vel)。与 predict_ball_trajectory 一致。"""
    p = np.asarray(pos, dtype=np.float64).copy()
    v = np.asarray(vel, dtype=np.float64).copy()
    steps = int(round(duration / dt))
    for _ in range(steps):
        a = _accel(v, k)
        v = v + a * dt
        p = p + v * dt
    return p, v


def _sample_positions(pos: np.ndarray, vel: np.ndarray, k: float,
                      t_rel: np.ndarray, dt: float = 0.002) -> np.ndarray:
    """从 (pos, vel) 出发积分，返回 t_rel（相对首帧）各时刻的位置 (N,3)。"""
    traj, ts = predict_ball_trajectory(pos, vel, k=k, dt=dt,
                                       horizon=float(t_rel[-1]) + dt)
    out = np.empty((len(t_rel), 3))
    for ax in range(3):
        out[:, ax] = np.interp(t_rel, ts, traj[:, ax])
    return out


def fit_ball_state(t_obs: np.ndarray, p_obs: np.ndarray, k: float = 0.0,
                   t_eval: float | None = None,
                   refine_k: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """对含噪观测位置做弹道(+阻力)拟合，返回 (pos, vel, k_est) 于 t_eval 时刻。

    Args:
        t_obs: (N,) 绝对时间（秒）；p_obs: (N,3) 观测位置；k: 阻力系数先验。
        t_eval: 求值时刻（秒），默认最新观测 t_obs[-1]。
        refine_k: True（默认）= Gauss-Newton 联合细化 7 参数 (p0, v0, k)，
            返回在线估计的 k_est；False = 固定入参 k，只细化 6 参数
            (p0, v0)（与 6a1f3d5 原版算法逐行等价），k_est == 入参 k。

    先用 k=0 闭式二次拟合初始化 (p0, v0)；refine_k=True 且 k>0 时用
    Gauss-Newton 同时细化 (p0, v0, k)（前向模型复用 predict_ball_trajectory，
    7 参数有限差分雅可比）。返回的 k_est 是观测窗内辨识出的真实阻力系数——
    长滞空下阻力失配会放大成拦截点预测偏差，固定先验 k 无法适应域随机化的
    真实 k，必须在线估计。refine_k=False 用于短观测窗 + 高噪声场景（如 V1
    的 σ=5mm + ≤16 帧窗）：此时 7 参数联合拟合病态（v0 吸收阻力误差，
    末段按 k_est 积分进一步污染状态），改用固定先验 k 拟合与积分。
    """
    t_obs = np.asarray(t_obs, dtype=np.float64).reshape(-1)
    p_obs = np.asarray(p_obs, dtype=np.float64)
    n = t_obs.shape[0]
    if n < 4:
        raise ValueError("fit_ball_state needs >= 4 observations")
    t0 = t_obs[0]
    t_rel = t_obs - t0
    if t_rel[-1] <= 0:
        return p_obs[-1].copy(), np.zeros(3), float(k)

    # 1) 初始化：k=0 逐轴二次拟合（p(t) = p0 + v0*t + 0.5*a*t^2）
    A = np.column_stack([np.ones(n), t_rel, t_rel * t_rel])
    coef, *_ = np.linalg.lstsq(A, p_obs, rcond=None)  # (3,3)
    p0 = coef[0].copy()
    v0 = coef[1].copy()

    # 2) k>0 Gauss-Newton 细化
    k_est = float(k)
    if k > 0 and refine_k:
        # 联合细化 (p0, v0, k) 7 参数
        pred = _sample_positions(p0, v0, k_est, t_rel)
        for _ in range(6):
            resid = (pred - p_obs).ravel()
            base = np.concatenate([p0, v0, [k_est]])
            J = np.empty((3 * n, 7))
            for j in range(7):
                pert = base.copy()
                h = 1e-3 if j == 6 else 1e-6
                pert[j] += h
                pred_p = _sample_positions(pert[:3], pert[3:6], pert[6], t_rel)
                J[:, j] = (pred_p - pred).ravel() / h
            delta, *_ = np.linalg.lstsq(J.T @ J + 1e-9 * np.eye(7),
                                       -J.T @ resid, rcond=None)
            p0 = p0 + delta[:3]
            v0 = v0 + delta[3:6]
            k_est = float(np.clip(k_est + delta[6], 0.0, 0.05))
            pred = _sample_positions(p0, v0, k_est, t_rel)
            if np.linalg.norm(delta) < 1e-7:
                break
    elif k > 0:
        # 固定 k，只细化 (p0, v0) 6 参数——与 6a1f3d5 原版逐行等价
        # （迭代次数 3 / eps 1e-6 / 正则 1e-9 / 收敛阈值 1e-6 全部照旧，
        # 位级复现关键，勿改）
        pred = _sample_positions(p0, v0, k, t_rel)
        for _ in range(3):
            resid = (pred - p_obs).ravel()
            base = np.concatenate([p0, v0])
            eps = 1e-6
            J = np.empty((3 * n, 6))
            for j in range(6):
                pert = base.copy()
                pert[j] += eps
                pred_p = _sample_positions(pert[:3], pert[3:], k, t_rel)
                J[:, j] = (pred_p - pred).ravel() / eps
            delta, *_ = np.linalg.lstsq(J.T @ J + 1e-9 * np.eye(6),
                                       -J.T @ resid, rcond=None)
            p0 = p0 + delta[:3]
            v0 = v0 + delta[3:]
            pred = _sample_positions(p0, v0, k, t_rel)
            if np.linalg.norm(delta) < 1e-6:
                break

    # 3) 评估到 t_eval
    if t_eval is None:
        t_eval = t_obs[-1]
    dur = float(t_eval) - t0
    if dur <= 0:
        return p0.copy(), v0.copy(), k_est
    pos, vel = _integrate_state(p0, v0, k_est, 0.002, dur)
    return pos, vel, k_est
