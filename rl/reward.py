"""RL 阶段 1/2 · reward（host 侧从 StepTrace 逐步计算，numpy）。

PPO 的 reward 是逐步（per-timestep）的，而命中/落点等终局信息在 trace 里以逐步
接触/位姿形式存在，故本模块把 `env.rollout` 返回的 trace（已 swap 为
(B, N, ...)）、final 与 extras 转成 (B, N) 的逐步 reward 与 done。

reward 五项（权重 DEFAULT_WEIGHTS，对应 D4 + 阶段 2 落点扩展）：
1. r_close：挥拍峰值窗（|swing_f - onset - peak_local| <= PEAK_WIN，与
   _reduce_batch 的 min_dist 同窗口）内，-close_w * ||ball - site||——
   引导腕部把拍面贴近球（击球时刻 site-ball 距越小越好）。
2. r_hit：终局命中奖励 +contact_w，一次性摊在「最后一个 active 步」上
   （用 final.hit 而非逐步 contact 累加，避免接触时长耦合放大稀疏奖励）。
3. r_landing = -landing_w * min(||落地-目标||, LAND_DIST_CAP)（均匀截断，无断崖）。
4. r_clear：终局过网高度项 clear_w*tanh((z_cross-NET_HEIGHT)/CLEAR_SCALE)，z_cross 为球首个向上穿越 x=NET_X 的插值 z；未过网回退 -clear_w（tanh 饱和）。
5. r_act：动作正则 -act_w * ||δ||^2——抑制过量腕部动作。

关键掩码（吸收态）：env 是定长 scan + done 吸收态，done 后 trace 仍记录。
- 所有 reward 仅在 active=True 的步非零；done 后一律 0。
- 绝不计算窗外/未激活步的距离（球 park 于 (100,100,5) 时 |ball-site|~100m，
  漏掩码会让 value head 直接爆炸）。
- done = ~active 供 GAE 在 episode 边界 bootstrap=0。
"""

from __future__ import annotations

import numpy as np

# 对齐 rl/tennis_env.py 的常量（本地声明避免 import 重模块）
MODE_SWING = 3
PEAK_WIN = 10
NET_X = 3.0                # 球网 x 坐标（scene_tennis.xml <body name="net" pos="3 0 0">）
NET_HEIGHT = 0.914         # 网高（= tennis_env.NET_HEIGHT；过网 clearance 参考）
GROUND_Z = 0.05            # 球落地判定阈值（= tennis_env.GROUND_Z）
CLEAR_SCALE = 0.15         # 过网高度 tanh 线性化尺度（m）：阈值处梯度 = clear_w/CLEAR_SCALE = 20/m
LAND_DIST_CAP = 6.0        # 落点距离上限（m）：去掉 -10 断崖，改均匀截断

DEFAULT_WEIGHTS = {"contact_w": 5.0, "close_w": 1.0, "act_w": 0.1,
                   "landing_w": 1.0, "clear_w": 3.0}


def landing_xy(ball: np.ndarray, ground_z: float = GROUND_Z) -> np.ndarray:
    """球落地位置 (B, 2)：z 首次 < ground_z 的 (x, y)；整回合未落地的用最后一帧。"""
    B, N, _ = ball.shape
    z = ball[..., 2]
    below = z < ground_z
    any_below = below.any(axis=-1)
    idx = np.argmax(below, axis=-1)               # 无 below 时 = 0
    safe = np.where(any_below, idx, N - 1)
    return np.stack([ball[np.arange(B), safe, 0],
                     ball[np.arange(B), safe, 1]], axis=-1)


def net_cross_z(ball: np.ndarray, active: np.ndarray, net_x: float = NET_X):
    """首个向上穿越 x=net_x 的球心 z（线性插值）+ 是否穿越。

    ball (B,N,3) 步后球心, active (B,N)。向下（来球）穿越与 inactive 步
    （park 于 x=100）均被排除；未穿越回退 z=0，crossed=False。
    """
    B, N, _ = ball.shape
    x = ball[..., 0]
    xa = np.where(active, x, -np.inf)
    up = (xa[:, :-1] <= net_x) & (xa[:, 1:] > net_x)
    crossed = up.any(axis=1)
    k = np.argmax(up, axis=1)
    rows = np.arange(B)
    x0, x1 = x[rows, k], x[rows, k + 1]
    z0, z1 = ball[rows, k, 2], ball[rows, k + 1, 2]
    denom = np.where(np.abs(x1 - x0) < 1e-9, 1.0, x1 - x0)
    frac = (net_x - x0) / denom
    zc = (z0 + frac * (z1 - z0)).astype(np.float32)
    return np.where(crossed, zc, 0.0).astype(np.float32), crossed


def compute_rewards(env, tr_np: dict, fin_np: dict, extras: dict,
                    contact_w: float = 5.0, close_w: float = 1.0,
                    act_w: float = 0.1, landing_w: float = 1.0,
                    clear_w: float = 3.0):
    """从单批 trace/final/extras（numpy，(B, N, ...)）算逐步 reward 与 done。

    Args:
        env: 含 swing_end / onset / peak_local 的对象（TennisEnv 或等价）。
        tr_np: env.rollout 返回的 trace 转 numpy 后的 dict；需含
            active/mode/swing_f/ball_pos/site_pos/delta。
        fin_np: env.rollout 返回的 final 转 numpy 后的 dict；需含 hit (B,)。
        extras: env.reset 返回的 extras 转 numpy 后的 dict；需含 target_landing (B, 2)。

    Returns:
        (rewards (B, N) float32, dones (B, N) bool)。
    """
    active = tr_np["active"]                       # (B, N)
    mode = tr_np["mode"]                           # (B, N)
    swing_f = tr_np["swing_f"]                     # (B, N)
    ball = tr_np["ball_pos"]                       # (B, N, 3)
    site = tr_np["site_pos"]                       # (B, N, 3)
    delta = tr_np["delta"]                         # (B, N, 3)
    hit = fin_np["hit"]                            # (B,)

    # 挥拍段（与 env._reduce_batch 的 swing 谓词一致）+ 峰值窗
    swing = active & (mode == MODE_SWING) & (swing_f <= env.swing_end)
    in_win = swing & (np.abs((swing_f - env.onset) - env.peak_local) <= PEAK_WIN)

    dist = np.linalg.norm(ball - site, axis=-1)    # (B, N)
    # np.where 值选择而非乘法掩码：窗外 dist 若为 inf/NaN，0*inf=NaN 会漏进 return
    r_close = -close_w * np.where(in_win, dist, 0.0)
    r_act = -act_w * np.sum(delta * delta, axis=-1)
    rewards = r_close + r_act

    # 终局奖励：命中 / 落点 / 过网，摊在每个 env 的最后一个 active 步
    B, N = rewards.shape
    last_active = N - 1 - np.argmax(active[:, ::-1], axis=1)   # (B,)
    rows = np.arange(B)

    rewards[rows, last_active] += contact_w * hit.astype(np.float32)

    landing = landing_xy(ball)                      # (B, 2)
    target_landing = np.asarray(extras["target_landing"])   # (B, 2)
    land_dist = np.linalg.norm(landing - target_landing, axis=-1)  # (B,)
    r_landing = -landing_w * np.minimum(land_dist, LAND_DIST_CAP)
    z_cross, crossed = net_cross_z(ball, active)    # (B,), (B,)
    clearance = np.where(crossed, z_cross - NET_HEIGHT, -3.0 * CLEAR_SCALE)
    r_clear = (clear_w * np.tanh(clearance / CLEAR_SCALE)).astype(np.float32)
    rewards[rows, last_active] += r_landing + r_clear

    rewards = np.where(active, rewards, 0.0)
    return rewards.astype(np.float32), ~active
