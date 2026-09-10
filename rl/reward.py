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
3. r_landing：终局落点奖励，-landing_w * ||落地位置 - 目标落点||（过网球）；
   下网球给大惩罚（-landing_w * LANDING_PENALTY）——引导球不仅打中、还打到目标
   落点（阶段 2「命中→落点」核心）。
4. r_net：终局过网奖励，过网（落地 x > NET_X）+net_w，下网 -net_w——引导球过网。
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
GROUND_Z = 0.05            # 球落地判定阈值（= tennis_env.GROUND_Z）
LANDING_PENALTY = 20.0     # 下网球等效落点距离（惩罚用）

DEFAULT_WEIGHTS = {"contact_w": 5.0, "close_w": 1.0, "act_w": 0.1,
                   "landing_w": 0.5, "net_w": 1.0}


def _landing_xy(ball: np.ndarray, ground_z: float = GROUND_Z) -> np.ndarray:
    """球落地位置 (B, 2)：z 首次 < ground_z 的 (x, y)；整回合未落地的用最后一帧。"""
    B, N, _ = ball.shape
    z = ball[..., 2]
    below = z < ground_z
    any_below = below.any(axis=-1)
    idx = np.argmax(below, axis=-1)               # 无 below 时 = 0
    safe = np.where(any_below, idx, N - 1)
    return np.stack([ball[np.arange(B), safe, 0],
                     ball[np.arange(B), safe, 1]], axis=-1)


def compute_rewards(env, tr_np: dict, fin_np: dict, extras: dict,
                    contact_w: float = 5.0, close_w: float = 1.0,
                    act_w: float = 0.1, landing_w: float = 0.5,
                    net_w: float = 1.0):
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

    landing = _landing_xy(ball)                     # (B, 2)
    target_landing = np.asarray(extras["target_landing"])   # (B, 2)
    land_dist = np.linalg.norm(landing - target_landing, axis=-1)  # (B,)
    over_net = landing[:, 0] > NET_X               # 落地 x 越过网 = 过网
    r_landing = -landing_w * np.where(over_net, land_dist, LANDING_PENALTY)
    r_net = net_w * np.where(over_net, 1.0, -1.0)
    rewards[rows, last_active] += r_landing + r_net

    rewards = np.where(active, rewards, 0.0)
    return rewards.astype(np.float32), ~active
