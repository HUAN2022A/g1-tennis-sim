"""RL T3 单测：rl/tennis_env.py 批量 MJX 网球环境（固定种子，不依赖长仿真）。

覆盖：reset 采样分布（发射可行域/T 网格/DR 范围/onset 复位）、单步物理
合理性（PD 保持 / 球待命 / 有限性）、模式转移（IDLE→TRACK 发射步跳变）、
racket-ball 接触检测（真值接触谓词）、read_fit 就绪门槛。

运行：uv run python tests/test_rl_env.py
前置：storage/ckpts/rl/v1_calib.npz（uv run python rl/tennis_env.py --calibrate）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco.mjx as mjx  # noqa: E402

import rl.tennis_env as te  # noqa: E402

print(f"[env] constructing TennisEnv(num_envs=8)（加载缓存标定 + 参考轨迹 + jit）...")
ENV = te.TennisEnv(num_envs=8, n_steps=16)   # 全部测试共用（编译一次）
B = ENV.num_envs


# ---------------------------------------------------------------------------
# reset 采样分布
# ---------------------------------------------------------------------------

def test_reset_distribution():
    state, extras = ENV.reset(seed=0)
    target = np.asarray(extras["serve_target"])
    T = np.asarray(extras["T_flight"])
    k = np.asarray(extras["k_drag"])
    launch = np.asarray(extras["launch_step"])
    p_swing = np.asarray(ENV.p_swing)

    # 拦截目标落在 P_swing 圆盘（水平 <= feas_radius，z ±0.02）
    dev = np.linalg.norm(target - p_swing, axis=-1)
    assert np.all(dev <= te.FEAS_RADIUS + 1e-4), dev
    assert np.all(np.abs(target[:, 2] - p_swing[2]) <= 0.02 + 1e-5)
    assert dev.mean() > 0.01, "固定种子下采样应铺开圆盘（非退化）"

    # 飞行时间：帧网格量化 + clip
    assert np.all((T >= te.FLIGHT_LO - 0.02 - 1e-6)
                  & (T <= te.FLIGHT_HI + 0.02 + 1e-6)), T
    grid_err = np.abs((T - 0.26) / 0.02 - np.round((T - 0.26) / 0.02))
    assert np.all(grid_err < 1e-4), grid_err

    # DR 范围
    assert np.all((k >= te.DR_K_LO - 1e-6) & (k <= te.DR_K_HI + 1e-6)), k
    m = np.asarray(state.ball_mass)
    base = ENV.base_ball_mass
    assert np.all((m >= 0.95 * base - 1e-6) & (m <= 1.05 * base + 1e-6)), m
    fr = np.asarray(state.ball_fric)
    assert np.all((fr >= 0.4 - 1e-5) & (fr <= 0.7 + 1e-5)), fr

    # launch_step = max(1, 30 - int((T - t_lead)*50))
    expect = np.maximum(1, 30 - ((T - ENV.t_lead) * 50).astype(np.int32))
    assert np.array_equal(launch, expect), (launch, expect)

    # 机器人复位 = ref_traj[onset]，root 速度清零，球 park
    qpos = np.asarray(state.data.qpos)
    ref_onset = np.asarray(ENV.ref_qpos[ENV.onset])
    assert np.allclose(qpos[:, :36], ref_onset[None], atol=1e-6)
    assert np.allclose(np.asarray(state.data.qvel), 0.0, atol=1e-7)
    assert np.allclose(qpos[:, ENV.ball_adr:ENV.ball_adr + 3],
                       np.asarray(te.PARK_POS)[None], atol=1e-5)
    assert np.all(np.asarray(state.mode) == te.MODE_IDLE)
    assert np.all(np.asarray(state.trig_step) == -1)
    assert np.allclose(np.asarray(state.last_action), 0.0)
    print(f"  reset: target_dev mean {dev.mean():.3f}m, T∈[{T.min():.2f},{T.max():.2f}]"
          f", k∈[{k.min():.4f},{k.max():.4f}], launch∈[{launch.min()},{launch.max()}]")


# ---------------------------------------------------------------------------
# 单步物理合理性
# ---------------------------------------------------------------------------

def test_step_physics_sanity():
    state, _ = ENV.reset(seed=1)
    state, tr = ENV.step(state)          # IDLE 步（发射前）
    qpos = np.asarray(state.data.qpos)
    qvel = np.asarray(state.data.qvel)
    assert np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel))
    assert np.all(np.asarray(tr.mode) == te.MODE_IDLE)
    assert np.all(np.asarray(tr.active))
    # 机器人 PD 保持 onset 姿态（20ms 内偏差有限）
    ref_onset = np.asarray(ENV.ref_qpos[ENV.onset])
    jerr = np.abs(qpos[:, 7:36] - ref_onset[None, 7:36]).max()
    assert jerr < 0.3, f"joint drift after 1 ctrl step: {jerr}"
    # 球仍近似 park（自由落体 20ms 位移 ~2mm）
    bz = qpos[:, ENV.ball_adr + 2]
    assert np.all((bz > 4.9) & (bz <= 5.0 + 1e-6)), bz
    # 球四元数保持归一
    q = qpos[:, ENV.ball_adr + 3:ENV.ball_adr + 7]
    assert np.allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-5)
    print(f"  1 ctrl step: joint drift max {jerr:.4f} rad, ball z∈[{bz.min():.3f},"
          f"{bz.max():.3f}], quat norms ok")


# ---------------------------------------------------------------------------
# 模式转移：IDLE --(launch_step)--> TRACK，球瞬移到发射点
# ---------------------------------------------------------------------------

def test_mode_transition_launch():
    state, extras = ENV.reset(seed=2)
    launch = np.asarray(extras["launch_step"])
    serve_pos = np.asarray(extras["serve_pos"])
    L = int(launch.max())
    modes = []
    ball_xyz = []
    for _ in range(L + 2):
        state, tr = ENV.step(state)
        modes.append(np.asarray(tr.mode))
        ball_xyz.append(np.asarray(tr.ball_pos).copy())
    modes = np.stack(modes)               # (L+2, B)
    ball_xyz = np.stack(ball_xyz)         # (L+2, B, 3)
    for e in range(B):
        le = int(launch[e])
        assert (modes[:le, e] == te.MODE_IDLE).all(), (e, le, modes[:le + 2, e])
        assert (modes[le:le + 2, e] == te.MODE_TRACK).all(), (e, modes[:le + 2, e])
        # 发射步球从 park 跳到发射点（首个 TRACK 步的步后位置接近 serve_pos）
        d0 = np.linalg.norm(ball_xyz[le, e] - np.asarray(te.PARK_POS))
        d1 = np.linalg.norm(ball_xyz[le, e] - serve_pos[e])
        assert d0 > 10.0 and d1 < 1.0, (e, d0, d1)
    print(f"  launch: IDLE→TRACK at launch_step（各环境 "
          f"{launch.min()}..{launch.max()}），球瞬移到发射点")


# ---------------------------------------------------------------------------
# 接触检测谓词（真值：球放到拍面中心 → 接触）
# ---------------------------------------------------------------------------

def test_contact_predicate():
    state, _ = ENV.reset(seed=3)
    fwd = jax.jit(jax.vmap(lambda d: mjx.forward(ENV.mx_model, d)))
    collide = jax.jit(ENV._racket_ball_contact)
    site = np.asarray(state.data.site_xpos[:, ENV.site_id])
    qpos = np.asarray(state.data.qpos).copy()
    # 球放拍面中心 → 全接触
    qpos[:, ENV.ball_adr:ENV.ball_adr + 3] = site
    d2 = fwd(state.data.replace(qpos=jnp.asarray(qpos, dtype=jnp.float32)))
    assert np.asarray(collide(d2)).all()
    # 偏移 1m → 不接触
    qpos[:, ENV.ball_adr:ENV.ball_adr + 3] = site + np.array([1.0, 0.0, 0.0])
    d3 = fwd(state.data.replace(qpos=jnp.asarray(qpos, dtype=jnp.float32)))
    assert not np.asarray(collide(d3)).any()
    print("  racket-ball 接触谓词：球在拍面中心 8/8 True，偏移 1m 8/8 False")


# ---------------------------------------------------------------------------
# read_fit 就绪门槛（观测窗 >= 4 才可用——V1 决策等滑窗填满语义）
# ---------------------------------------------------------------------------

def test_read_fit_ready_gate():
    buf = ENV.buf
    st = buf.make_init((B,))
    ready_seq = []
    for i in range(6):
        # 确定性注入（零噪声、永不 drop；step 为标量——批量缓冲单时间轴）
        st = buf.update_with_draws(st, jnp.full((B, 3), 1.0 + 0.001 * i),
                                   i, jnp.zeros((B, 3)), jnp.ones((B,)))
        _, _, _, ready = buf.read_fit(st, i, k=te.PRED_DRAG_K,
                                      min_points=te.OBS_WINDOW,
                                      max_points=te.MAX_POINTS)
        ready_seq.append(bool(np.asarray(ready[0])))
    # 延迟 1 帧 + min_points=4：read(i) 的 cutoff=i-1 → 可用观测 i 个
    #（标签 0..i-1），i>=4 才就绪（V1：决策等 4 点滑窗填满的同一语义）
    assert ready_seq == [False, False, False, False, True, True], ready_seq
    print(f"  read_fit ready 门：{ready_seq}（i>=4 就绪，与 numpy 一致）")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"[{name}]")
            fn()
            print(f"PASS {name}")
    print("all tests passed")
