"""走到目标点的简单位置控制器（V3 用）：把 (root_xy, root_yaw, target_xy)
转成 walk policy 的 cmd_vel = [lin_x, lin_y, ang_yaw]。

walk policy（deploy/walk_policy.py）是开环速度指令（前进/侧移/偏航），
无位置伺服；本控制器补上"转向目标 → 前进 → 到达归零"的最小闭环。
纯 numpy，可脱离 MuJoCo 单测。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _wrap_angle(a: float) -> float:
    """把角度归一到 [-pi, pi)。"""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class GotoController:
    k_lin: float = 1.0          # 前进增益（1/s 量级）
    k_yaw: float = 1.5          # 偏航增益
    v_max: float = 0.35         # 前进速度上限（对齐 deploy/constants.KEYBOARD_MAX_SPEED）
    w_max: float = 0.5          # 偏航角速度上限（rad/s）
    turn_thresh: float = 0.25   # 朝向误差超过此值（rad）先原地转向
    arrive_dist: float = 0.20   # 到达阈值（m）

    def cmd_vel(self, root_xy, root_yaw, target_xy) -> np.ndarray:
        """返回 [lin_x, lin_y, ang_yaw]（机器人体坐标系）。

        先原地转向对齐目标方向（朝向误差 > turn_thresh），再前进 + 比例偏航
        微调；进入 arrive_dist 后归零。全程不用侧移（lin_y=0）。
        """
        root_xy = np.asarray(root_xy, dtype=np.float64)
        target_xy = np.asarray(target_xy, dtype=np.float64)
        delta = target_xy - root_xy
        dist = float(np.linalg.norm(delta))
        if dist < self.arrive_dist:
            return np.zeros(3)

        target_bearing = float(np.arctan2(delta[1], delta[0]))
        yaw_err = _wrap_angle(target_bearing - float(root_yaw))

        if abs(yaw_err) > self.turn_thresh:
            w = float(np.clip(self.k_yaw * yaw_err, -self.w_max, self.w_max))
            return np.array([0.0, 0.0, w])

        v = float(np.clip(self.k_lin * dist, 0.0, self.v_max))
        w = float(np.clip(self.k_yaw * yaw_err, -self.w_max, self.w_max))
        return np.array([v, 0.0, w])
