"""网球工具纯函数单测（fit_ball_state / select_clip / GotoController）。

运行：uv run python tests/test_tennis.py
"""

import numpy as np

from utils.ball_predict import fit_ball_state, predict_ball_trajectory, _integrate_state
from utils.tennis_clips import ClipSpec, select_clip
from utils.goto import GotoController


def test_fit_ball_state_reduces_velocity_error():
    rng = np.random.default_rng(0)
    k = 0.0125
    p0 = np.array([0.0, 0.0, 3.0])
    v0 = np.array([3.0, 0.5, 2.0])
    dt = 0.02
    N = 16
    traj, ts = predict_ball_trajectory(p0, v0, k=k, dt=0.002, horizon=(N - 1) * dt + 0.01)
    t_obs = np.arange(N) * dt
    P_true = np.array([[np.interp(t, ts, traj[:, a]) for a in range(3)] for t in t_obs])
    P_obs = P_true + rng.uniform(-0.005, 0.005, size=P_true.shape)
    _, v_fit, _ = fit_ball_state(t_obs, P_obs, k=k)
    _, v_end = _integrate_state(p0, v0, k, 0.002, (N - 1) * dt)
    # 4 帧 LSQ 斜率基线（速度在窗口中点，含阻力减速失配）
    t4 = t_obs[-4:]
    P4 = P_obs[-4:]
    tc = t4 - t4.mean()
    v_lsq = (tc[:, None] * (P4 - P4.mean(axis=0))).sum(axis=0) / (tc ** 2).sum()
    _, v_mid = _integrate_state(p0, v0, k, 0.002, (t4[0] + t4[-1]) / 2)
    err_fit = np.linalg.norm(v_fit - v_end)
    err_lsq = np.linalg.norm(v_lsq - v_mid)
    assert err_fit < err_lsq, f"fit {err_fit:.4f} should beat LSQ {err_lsq:.4f}"


def test_fit_ball_state_k0_exact():
    p0 = np.array([0.0, 0.0, 1.0])
    v0 = np.array([1.0, 0.0, 0.0])
    tb = np.arange(8) * 0.02
    Pb = np.array([p0 + v0 * t + np.array([0, 0, -0.5 * 9.81 * t * t]) for t in tb])
    _, vf, _ = fit_ball_state(tb, Pb, k=0.0)
    v_true_end = v0 + np.array([0, 0, -9.81]) * tb[-1]
    assert np.linalg.norm(vf - v_true_end) < 1e-8


def test_select_clip_side():
    fh = ClipSpec(path="fh.npz", side="forehand")
    bh = ClipSpec(path="bh.npz", side="backhand")
    clips = [fh, bh]
    assert select_clip(np.array([0.0, 0.5, 1.0]), 0.0, clips, -1.0) is fh
    assert select_clip(np.array([0.0, -0.5, 1.0]), 0.0, clips, -1.0) is bh
    # 符号翻转后左右互换
    assert select_clip(np.array([0.0, 0.5, 1.0]), 0.0, clips, +1.0) is bh


def test_goto_turn_then_approach():
    gc = GotoController()
    # 朝向目标正前方，直接前进，不偏航
    v = gc.cmd_vel([0.0, 0.0], 0.0, [1.0, 0.0])
    assert v[0] > 0 and v[2] == 0.0
    # 目标在身后（yaw_err = pi），先原地转向
    v = gc.cmd_vel([0.0, 0.0], 0.0, [-1.0, 0.0])
    assert v[0] == 0.0 and v[2] != 0.0
    # 到达阈值内归零
    v = gc.cmd_vel([0.0, 0.0], 0.0, [0.1, 0.0])
    assert np.all(v == 0.0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
