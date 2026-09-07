"""网球观测模块：位置噪声 + 帧级 dropout + 延迟 + 4 帧滑窗速度。

按 docs/TENNIS_STATE_MACHINE_PLAN.md 第 3.1c 节与 LATENT 论文 4.2 节实现：
- 球位置加均匀噪声（|noise| <= sigma，模拟检测误差）
- 帧级 dropout：以概率 p 丢弃本帧观测（沿用上一次有效观测）
- 延迟：观测滞后 delay 帧可用
- 速度：4 帧滑窗平均（位置差分对噪声极敏感，LATENT 用同样的平滑方式）

V0 不用本模块（直接真值）；V1 起作为预测模块的输入接口。
真值 → BallObserver.update() → BallObserver.read() = 观测值（预测用）。
"""

from __future__ import annotations

from collections import deque

import numpy as np


class BallObserver:
    """球状态观测器：真值进、带噪观测出。

    Args:
        pos_sigma: 位置噪声幅度（米，均匀分布半宽，各轴独立）
        dropout_p: 帧级 dropout 概率（0 = 不丢帧）
        delay_frames: 观测延迟（帧数）
        smooth_window: 速度滑窗长度（LATENT 用 4）
        seed: 随机种子（可复现）
    """

    def __init__(self, pos_sigma: float = 0.005, dropout_p: float = 0.05,
                 delay_frames: int = 1, smooth_window: int = 4, seed: int = 0):
        self.pos_sigma = float(pos_sigma)
        self.dropout_p = float(dropout_p)
        self.delay_frames = int(delay_frames)
        self.smooth_window = max(int(smooth_window), 2)
        self._rng = np.random.default_rng(seed)

        # 历史缓冲：最近 (delay + smooth) 帧的带噪位置（含时间戳步号）
        self._hist: deque[tuple[int, np.ndarray]] = deque(
            maxlen=self.delay_frames + self.smooth_window + 2)
        self._last_valid: np.ndarray | None = None  # 最近一次未 drop 的观测

    def reset(self) -> None:
        self._hist.clear()
        self._last_valid = None

    def update(self, true_pos: np.ndarray, step: int) -> None:
        """喂入当前真值（每个控制步调用一次）。step 为控制步号。"""
        pos = np.asarray(true_pos, dtype=np.float64).copy()
        # 帧级 dropout：丢帧时沿用上一次有效观测
        if self._last_valid is not None and self._rng.random() < self.dropout_p:
            obs = self._last_valid.copy()
        else:
            obs = pos + self._rng.uniform(-self.pos_sigma, self.pos_sigma, size=3)
            self._last_valid = obs.copy()
        self._hist.append((step, obs))

    def read(self, step: int, ctrl_dt: float,
             min_points: int = 2) -> tuple[np.ndarray, np.ndarray] | None:
        """读取当前可用观测（已含延迟）：返回 (观测位置, 滑窗平均速度)。

        返回 None 表示尚无足够观测（启动初期 / min_points 未满足）。
        min_points: 做出决策（预测/触发/MISS 判定）所需的最少可用观测数；
        2 点估计的噪声过大（速度误差 ~2σ/dt），默认决策场景应传
        smooth_window 让调用方等窗口填满。
        速度 = 最近 n 个观测位置的最小二乘线性拟合斜率（用步号做时间轴）。
        """
        # 延迟：只允许使用 step - delay 时刻及以前的观测
        cutoff = step - self.delay_frames
        avail = [(s, p) for s, p in self._hist if s <= cutoff]
        if len(avail) < max(2, min_points):
            return None
        # 速度：最近 N 个观测的最小二乘斜率（用步号做时间轴）
        n = min(self.smooth_window, len(avail))
        seg = avail[-n:]
        t = np.asarray([s for s, _ in seg], dtype=np.float64) * ctrl_dt
        P = np.asarray([p for _, p in seg], dtype=np.float64)  # (n, 3)
        if t[-1] - t[0] < 1e-9:
            vel = np.zeros(3)
        else:
            # lstsq 斜率: v = sum((t-t̄)(P-P̄)) / sum((t-t̄)^2)
            tc = t - t.mean()
            vel = (tc[:, None] * (P - P.mean(axis=0))).sum(axis=0) / (tc ** 2).sum()
        return seg[-1][1].copy(), vel
