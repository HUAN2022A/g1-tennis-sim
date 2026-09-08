"""RL T2 单测：rl/ball_obs_jax 与 numpy 参照实现对齐（容差 1e-6）。

运行：uv run python tests/test_rl_ball.py

对齐策略：jax 与 numpy 随机数不同流——输入一律用 numpy 固定种子生成
（sample_serve 的随机 draw 先在 numpy 侧按 draw 顺序录制），同一份确定性
数据喂两版实现，而非比较同 seed 输出。
参照实现：utils/ball_predict.py、utils/ball_obs.py、utils/tennis_ball.py。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 仓库根（rl/ 不在 editable 安装映射内）

import numpy as np

import rl.ball_obs_jax as bj  # noqa: E402  不再隐式开启 x64——对齐测试显式 opt-in（下两行）

bj.enable_x64()  # noqa: E402  对齐测试需 float64（须先于任何 jnp 数组创建）

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import random as jrandom  # noqa: E402

from utils.ball_predict import (  # noqa: E402
    _accel,
    _integrate_state,
    fit_ball_state,
    predict_ball_trajectory,
    predict_intercept,
)
from utils.tennis_ball import design_serve_with_drag, sample_serve  # noqa: E402

TOL = 1e-6

assert jax.config.jax_enable_x64, "enable_x64() 后必须运行在 float64"


# ---------------------------------------------------------------------------
# 积分器
# ---------------------------------------------------------------------------

def test_accel_step_euler():
    """加速度与单步半隐式欧拉：同输入位级一致。"""
    for k in (0.0, 0.0125):
        v = np.array([3.0, -1.5, 2.0])
        a_np = _accel(v, k)
        a_j = np.asarray(bj.accel(jnp.asarray(v), k))
        d_a = float(np.max(np.abs(a_np - a_j)))
        assert d_a == 0.0, f"k={k} accel max|Δ|={d_a}"
        dt = 0.002
        p = np.array([1.0, 2.0, 3.0])
        v_np = v + a_np * dt
        p_np = p + v_np * dt
        p_j, v_j = bj.step_euler(jnp.asarray(p), jnp.asarray(v), k, dt)
        d_p = float(np.max(np.abs(p_np - np.asarray(p_j))))
        d_v = float(np.max(np.abs(v_np - np.asarray(v_j))))
        assert d_p == 0.0 and d_v == 0.0, f"k={k} step dP={d_p} dV={d_v}"
    print("  accel/step_euler max|Δ| = 0.0 (位级一致)")


def test_integrate_trajectory_alignment():
    """整条弹道（1501 点，k>0）与 predict_ball_trajectory 对齐。"""
    p0 = np.array([0.5, -0.2, 2.8])
    v0 = np.array([4.0, 1.0, 3.0])
    k = 0.0125
    P_np, t_np = predict_ball_trajectory(p0, v0, k=k)  # 默认 dt=0.002, horizon=3.0
    P_j, V_j, t_j = bj.integrate_trajectory(jnp.asarray(p0), jnp.asarray(v0), k)
    assert P_j.shape == (1501, 3) and t_j.shape == (1501,)
    d_p = float(np.max(np.abs(np.asarray(P_j) - P_np)))
    d_t = float(np.max(np.abs(np.asarray(t_j) - t_np)))
    assert d_p < TOL and d_t < TOL, f"traj dP={d_p} dt={d_t}"
    print(f"  integrate_trajectory max|Δpos|={d_p:.3e} max|Δt|={d_t:.3e}")
    # horizon 参数化路径
    P_j2, _, t_j2 = bj.integrate_trajectory(jnp.asarray(p0), jnp.asarray(v0), k,
                                            horizon=1.0, dt=0.002)
    P_np2, t_np2 = predict_ball_trajectory(p0, v0, k=k, horizon=1.0)
    assert np.asarray(P_j2).shape == P_np2.shape
    assert float(np.max(np.abs(np.asarray(P_j2) - P_np2))) < TOL


def test_integrate_state_dynamic_steps():
    """traced 步数（掩码扫描）与 _integrate_state 对齐。"""
    p0 = np.array([0.5, -0.2, 2.8])
    v0 = np.array([4.0, 1.0, 3.0])
    k = 0.0125
    for dur in (0.3, 0.02, 0.314):
        pe_np, ve_np = _integrate_state(p0, v0, k, 0.002, dur)
        pe_j, ve_j = bj.integrate_state(
            jnp.asarray(p0), jnp.asarray(v0), k,
            bj.duration_to_steps(dur), n_steps_max=256)
        d_p = float(np.max(np.abs(pe_np - np.asarray(pe_j))))
        d_v = float(np.max(np.abs(ve_np - np.asarray(ve_j))))
        assert d_p < TOL and d_v < TOL, f"dur={dur} dP={d_p} dV={d_v}"
        print(f"  integrate_state dur={dur}: max|Δpos|={d_p:.3e} max|Δvel|={d_v:.3e}")


def test_predict_intercept_alignment():
    """下降段穿越平面：插值位置与时刻对齐；不可达时 found=False。"""
    p0 = np.array([0.5, -0.2, 2.8])
    v0 = np.array([4.0, 1.0, 3.0])  # 先升后降
    for k in (0.0, 0.0125):
        p_np, t_np = predict_intercept(p0, v0, z_plane=1.5, k=k)
        p_j, t_j, found = bj.predict_intercept(jnp.asarray(p0), jnp.asarray(v0),
                                               1.5, k)
        assert bool(np.asarray(found))
        d_p = float(np.max(np.abs(p_np - np.asarray(p_j))))
        d_t = float(abs(t_np - np.asarray(t_j)))
        assert d_p < TOL and d_t < TOL, f"k={k} dP={d_p} dt={d_t}"
        print(f"  predict_intercept k={k}: max|Δpos|={d_p:.3e} |Δt|={d_t:.3e}")
    # 不可达（平面高于弹道顶点）：numpy (None, None) ↔ jax found=False
    p_np, t_np = predict_intercept(np.array([0.0, 0.0, 0.5]),
                                   np.array([1.0, 0.0, 5.0]), z_plane=10.0, k=0.0)
    assert p_np is None and t_np is None
    _, _, found = bj.predict_intercept(jnp.array([0.0, 0.0, 0.5]),
                                       jnp.array([1.0, 0.0, 5.0]), 10.0, 0.0)
    assert not bool(np.asarray(found))


# ---------------------------------------------------------------------------
# 弹道拟合（refine_k=False 固定 k）
# ---------------------------------------------------------------------------

def _make_noisy_window(seed=0, k=0.0125, n=16, sigma=0.005):
    """合成含噪观测窗（与 tests/test_tennis.py 同构造）。"""
    p0 = np.array([0.0, 0.0, 3.0])
    v0 = np.array([3.0, 0.5, 2.0])
    dt = 0.02
    traj, ts = predict_ball_trajectory(p0, v0, k=k, dt=0.002,
                                       horizon=(n - 1) * dt + 0.01)
    t_obs = np.arange(n) * dt
    P_true = np.array([[np.interp(t, ts, traj[:, a]) for a in range(3)]
                       for t in t_obs])
    rng = np.random.default_rng(seed)
    P_obs = P_true + rng.uniform(-sigma, sigma, size=P_true.shape)
    return t_obs, P_obs


def test_fit_ball_state_fixed_k_alignment():
    """合成含噪弹道拟合 vs numpy refine_k=False：(pos, vel) max|Δ| < 1e-6。"""
    k = 0.0125
    t_obs, P_obs = _make_noisy_window(seed=0, k=k)
    pos_np, vel_np, k_np = fit_ball_state(t_obs, P_obs, k=k, refine_k=False)
    pos_j, vel_j, k_j = bj.fit_ball_state_fixed_k(jnp.asarray(t_obs),
                                                  jnp.asarray(P_obs), k=k,
                                                  n_steps_max=512)
    assert float(np.asarray(k_j)) == k == k_np  # 固定 k：k_est 恒等于入参
    d_p = float(np.max(np.abs(pos_np - np.asarray(pos_j))))
    d_v = float(np.max(np.abs(vel_np - np.asarray(vel_j))))
    assert d_p < TOL and d_v < TOL, f"fit dP={d_p} dV={d_v}"
    print(f"  fit_ball_state_fixed_k: max|Δpos|={d_p:.3e} max|Δvel|={d_v:.3e}")

    # t_eval 显式前推（compensate_delay 用）
    pos_np2, vel_np2, _ = fit_ball_state(t_obs, P_obs, k=k, t_eval=0.31,
                                         refine_k=False)
    pos_j2, vel_j2, _ = bj.fit_ball_state_fixed_k(jnp.asarray(t_obs),
                                                  jnp.asarray(P_obs), k=k,
                                                  t_eval=0.31, n_steps_max=512)
    d_p2 = float(np.max(np.abs(pos_np2 - np.asarray(pos_j2))))
    d_v2 = float(np.max(np.abs(vel_np2 - np.asarray(vel_j2))))
    assert d_p2 < TOL and d_v2 < TOL, f"fit t_eval dP={d_p2} dV={d_v2}"
    print(f"  fit t_eval=0.31: max|Δpos|={d_p2:.3e} max|Δvel|={d_v2:.3e}")

    # k=0（跳过 GN 分支）
    t0_obs, P0_obs = _make_noisy_window(seed=1, k=0.0)
    pos_np3, vel_np3, _ = fit_ball_state(t0_obs, P0_obs, k=0.0, refine_k=False)
    pos_j3, vel_j3, _ = bj.fit_ball_state_fixed_k(jnp.asarray(t0_obs),
                                                  jnp.asarray(P0_obs), k=0.0,
                                                  n_steps_max=512)
    d_p3 = float(np.max(np.abs(pos_np3 - np.asarray(pos_j3))))
    d_v3 = float(np.max(np.abs(vel_np3 - np.asarray(vel_j3))))
    assert d_p3 < TOL and d_v3 < TOL, f"fit k=0 dP={d_p3} dV={d_v3}"
    print(f"  fit k=0（无 GN）: max|Δpos|={d_p3:.3e} max|Δvel|={d_v3:.3e}")


# ---------------------------------------------------------------------------
# 发球采样 / 阻力外力
# ---------------------------------------------------------------------------

def test_drag_force_alignment():
    """f = -k·m·v·|v| 对齐（含 k<=0 / 低速 两个守卫分支）。"""
    v = np.array([[3.0, -1.5, 2.0], [0.0, 0.0, 0.0], [1e-10, 0.0, 0.0]])
    k, m = 0.0125, 0.0577
    f_np = -k * m * v * np.linalg.norm(v, axis=-1, keepdims=True)
    f_j = np.asarray(bj.drag_force(jnp.asarray(v), k, m))
    d = float(np.max(np.abs(f_np - f_j)))
    assert d < TOL, f"drag d={d}"
    assert np.all(np.asarray(bj.drag_force(jnp.asarray(v), 0.0, m)) == 0.0)
    print(f"  drag_force max|Δ|={d:.3e}（k<=0 与 |v|<1e-9 守卫均返回 0）")


def test_design_serve_with_drag_alignment():
    """带阻力发射速度设计的 4 次迭代修正对齐。"""
    k = 0.0125
    for target, T in ((np.array([1.0, 0.05, 0.7]), 0.66),
                      (np.array([0.9, -0.2, 0.8]), 0.70),
                      (np.array([1.05, 0.1, 0.75]), 0.84)):
        p0_np, v_np = design_serve_with_drag(target, T, k, 1.0, 2.8)
        p0_j, v_j = bj.design_serve_with_drag(jnp.asarray(target), T, k, 1.0, 2.8)
        d_p = float(np.max(np.abs(p0_np - np.asarray(p0_j))))
        d_v = float(np.max(np.abs(v_np - np.asarray(v_j))))
        assert d_p < TOL and d_v < TOL, f"T={T} dP={d_p} dV={d_v}"
        print(f"  design_serve_with_drag T={T}: max|Δp0|={d_p:.3e} max|Δv|={d_v:.3e}")


def test_sample_serve_alignment():
    """录制 numpy sample_serve 的随机 draw 喂 jax 版：(p0, v, T, target) 对齐。"""
    p_swing = np.array([1.05, 0.1, 0.75])
    lo, hi = 0.65, 0.85
    for design_k in (0.0125, 0.0):
        seed = 7 if design_k > 0 else 11
        rng = np.random.default_rng(seed)
        p0_np, v_np, T_np, tg_np = sample_serve(p_swing, rng, 0.06, lo, hi,
                                                1.0, 2.8, design_k=design_k)
        # 录制 draw（顺序：random×2 → uniform(-0.02,0.02) → uniform(lo,hi)）
        twin = np.random.default_rng(seed)
        u_r, u_theta = twin.random(), twin.random()
        z = twin.uniform(-0.02, 0.02)
        T_raw = twin.uniform(lo, hi)
        # 录制自检：同公式重放应复现 sample_serve 输出（位级）
        r = 0.06 * np.sqrt(u_r)
        tg_rp = p_swing + np.array([r * np.cos(u_theta * 2 * np.pi),
                                    r * np.sin(u_theta * 2 * np.pi), z])
        T_rp = float(np.clip(0.26 + round((T_raw - 0.26) / 0.02) * 0.02,
                             lo - 0.02, hi + 0.02))
        assert np.max(np.abs(tg_rp - tg_np)) == 0.0 and T_rp == T_np
        # 均匀值反解回 U[0,1)（1 ulp 内，下游 < 1e-6 无虞）
        u_z01 = (z + 0.02) / 0.04
        u_T01 = (T_raw - lo) / (hi - lo)
        p0_j, v_j, T_j, tg_j = bj.sample_serve_batch(
            jnp.asarray(p_swing), jnp.asarray(u_r), jnp.asarray(u_theta),
            jnp.asarray(u_z01), jnp.asarray(u_T01), 0.06, lo, hi, 1.0, 2.8,
            design_k=design_k)
        d_p0 = float(np.max(np.abs(p0_np - np.asarray(p0_j))))
        d_v = float(np.max(np.abs(v_np - np.asarray(v_j))))
        d_T = float(abs(T_np - np.asarray(T_j)))
        d_tg = float(np.max(np.abs(tg_np - np.asarray(tg_j))))
        assert max(d_p0, d_v, d_T, d_tg) < TOL
        print(f"  sample_serve design_k={design_k}: max|Δp0|={d_p0:.3e} "
              f"max|Δv|={d_v:.3e} |ΔT|={d_T:.3e} max|Δtarget|={d_tg:.3e}")


# ---------------------------------------------------------------------------
# 球观测环形缓冲
# ---------------------------------------------------------------------------

def _drive_buffers(seed=3, n_steps=30, H=16, sigma=0.005, drop_p=0.3,
                   delay=1, ctrl_dt=0.02, mid_step=9):
    """numpy 镜像（deque 语义）与 jax 缓冲同源驱动。

    噪声/drop 判据由 numpy 固定种子生成后喂两版（jax 经 update_with_draws
    注入）。返回 (hist_full, hist_mid, st_full, st_mid, buf, meta)：
    mid 快照 = 前 mid_step+1 步（此时环形缓冲尚未覆盖旧槽，可测部分窗）。
    """
    p0 = np.array([0.0, 0.0, 3.0])
    v0 = np.array([3.0, 0.5, 2.0])
    k = 0.0125
    traj, ts = predict_ball_trajectory(p0, v0, k=k, dt=0.002, horizon=2.0)
    rng = np.random.default_rng(seed)
    steps = np.arange(n_steps)
    true_p = np.array([[np.interp(s * ctrl_dt, ts, traj[:, a]) for a in range(3)]
                       for s in steps])
    noises = rng.uniform(-sigma, sigma, size=(n_steps, 3))
    drop_us = rng.random(n_steps)

    hist = []
    hist_mid = None
    last_valid = None
    for i in range(n_steps):
        if last_valid is not None and drop_us[i] < drop_p:
            obs = last_valid.copy()  # 帧级 dropout：沿用上次有效观测
        else:
            obs = true_p[i] + noises[i]
            last_valid = obs.copy()
        hist.append((int(steps[i]), obs))
        if len(hist) > H:
            hist.pop(0)
        if i == mid_step:
            hist_mid = list(hist)  # 前 mid_step+1 条（尚未 pop）

    buf = bj.BallObsBuffer(hist_len=H, pos_sigma=sigma, dropout_p=drop_p,
                           delay_frames=delay, ctrl_dt=ctrl_dt)
    st = buf.make_init()
    st_mid = None
    for i in range(n_steps):
        st = buf.update_with_draws(st, jnp.asarray(true_p[i]), int(steps[i]),
                                   jnp.asarray(noises[i]),
                                   jnp.asarray(drop_us[i]))
        if i == mid_step:
            st_mid = st

    n_drop = int(sum(1 for i in range(1, n_steps) if drop_us[i] < drop_p))
    meta = {"k": k, "delay": delay, "ctrl_dt": ctrl_dt, "n_drop": n_drop,
            "hist_mid": hist_mid}
    return hist, st, st_mid, buf, meta


def _np_read_fit(hist, step, k, delay, ctrl_dt, min_points=8, max_points=16):
    """numpy 侧 read_fit 镜像（avail 过滤 + 尾部窗口 + fit refine_k=False）。"""
    cutoff = step - delay
    avail = [(s, p) for s, p in hist if s <= cutoff]
    if len(avail) < max(4, min_points):
        return None
    seg = avail[-max_points:]
    t = np.array([s for s, _ in seg]) * ctrl_dt
    P = np.array([p for _, p in seg])
    return fit_ball_state(t, P, k=k, refine_k=False)


def test_ball_obs_buffer_alignment():
    """缓冲历史（含 dropout/延迟/环形覆盖）与 read_fit 全链路对齐。"""
    H = 16
    hist, st, st_mid, buf, meta = _drive_buffers(H=H)
    k, delay, ctrl_dt = meta["k"], meta["delay"], meta["ctrl_dt"]
    assert meta["n_drop"] >= 3, "固定种子下应实际触发 dropout（否则测试空转）"

    # 历史内容位级一致（按步号排序后比对；环形覆盖后只剩最近 H 步）
    order = np.argsort(np.asarray(st.hist_step))
    steps_np = np.array([s for s, _ in hist])
    pos_np = np.array([p for _, p in hist])
    assert np.array_equal(np.asarray(st.hist_step)[order], steps_np)
    d_hist = float(np.max(np.abs(np.asarray(st.hist_pos)[order] - pos_np)))
    assert d_hist == 0.0, f"hist max|Δ|={d_hist}"
    print(f"  缓冲历史（{H} 槽环形 + {meta['n_drop']} 次 dropout）位级一致")

    # read_fit 对齐矩阵：
    #  (st, step, hist, max_points) —— 覆盖 15 点部分窗（环形缺最旧 1 点）、
    #  9 点部分窗（mid 快照 + 掩码填充）、6 点截窗、观测不足（numpy None）
    cases = [
        (st, 29, hist, 16, True),        # avail=15 (steps 14..28) < 16 → 填充掩码路径
        (st_mid, 9, meta["hist_mid"], 16, True),   # avail=9 → 填充掩码路径
        (st, 29, hist, 6, True),         # avail=15 → 截到最近 6 点
        (st_mid, 3, meta["hist_mid"], 16, False),  # avail=3 < max(4,8) → 不就绪
    ]
    for st_i, rd, hist_i, mp, expect_ready in cases:
        out = _np_read_fit(hist_i, rd, k, delay, ctrl_dt, min_points=8,
                           max_points=mp)
        pos_j, vel_j, k_j, ready = buf.read_fit(st_i, rd, k=k, min_points=8,
                                                max_points=mp)
        assert bool(np.asarray(ready)) == expect_ready, f"step={rd} mp={mp}"
        if out is None:
            assert np.all(np.asarray(pos_j) == 0.0)
            print(f"  read_fit step={rd} mp={mp}: numpy=None ↔ jax ready=False ✓")
            continue
        pos_npv, vel_npv, _ = out
        d_p = float(np.max(np.abs(pos_npv - np.asarray(pos_j))))
        d_v = float(np.max(np.abs(vel_npv - np.asarray(vel_j))))
        assert d_p < TOL and d_v < TOL, f"step={rd} mp={mp} dP={d_p} dV={d_v}"
        assert float(np.asarray(k_j)) == k
        n_pts = min(max(rd - delay + 1 - max(0, rd - H + 1), 0), mp)
        print(f"  read_fit step={rd} mp={mp}（{n_pts} 点窗）: "
              f"max|Δpos|={d_p:.3e} max|Δvel|={d_v:.3e}")


# ---------------------------------------------------------------------------
# jit / vmap / 批量
# ---------------------------------------------------------------------------

def test_jit_and_vmap():
    """全部核心函数 jit 编译通过；fit/intercept/design 支持批量（vmap 或显式 batch 维）。"""
    k = 0.0125
    p0 = np.array([0.5, -0.2, 2.8])
    v0 = np.array([4.0, 1.0, 3.0])

    # jit：积分器 / 拦截
    jtraj = jax.jit(bj.integrate_trajectory, static_argnames=("n_steps", "dt"))
    P, V, t = jtraj(jnp.asarray(p0), jnp.asarray(v0), k)
    assert P.shape == (1501, 3)
    jint = jax.jit(bj.integrate_state, static_argnames=("dt", "n_steps_max"))
    pe, ve = jint(jnp.asarray(p0), jnp.asarray(v0), k,
                  bj.duration_to_steps(0.3), n_steps_max=256)
    assert pe.shape == (3,)
    jpi = jax.jit(bj.predict_intercept, static_argnames=("n_steps", "dt"))
    p_i, t_i, found = jpi(jnp.asarray(p0), jnp.asarray(v0), 1.5, k)
    assert p_i.shape == (3,) and t_i.shape == ()

    # vmap：批量 fit vs numpy 逐样本
    B = 3
    t_b, P_b = [], []
    for s in range(B):
        t_obs, P_obs = _make_noisy_window(seed=s, k=k)
        t_b.append(t_obs)
        P_b.append(P_obs)
    fit_vm = jax.jit(jax.vmap(bj.fit_ball_state_fixed_k, in_axes=(0, 0, None)))
    pos_b, vel_b, k_b = fit_vm(jnp.asarray(np.stack(t_b)),
                               jnp.asarray(np.stack(P_b)), k)
    assert pos_b.shape == (B, 3)
    for s in range(B):
        pos_np, vel_np, _ = fit_ball_state(t_b[s], P_b[s], k=k, refine_k=False)
        d = max(float(np.max(np.abs(pos_np - np.asarray(pos_b[s])))),
                float(np.max(np.abs(vel_np - np.asarray(vel_b[s])))))
        assert d < TOL, f"vmap fit[{s}] d={d}"
        assert float(np.asarray(k_b[s])) == k
    print("  vmap(fit_ball_state_fixed_k) ×3 与 numpy 逐样本对齐（<1e-6）")

    # vmap：批量 predict_intercept
    pos_np_b = np.stack([p0, np.array([0.0, 0.0, 1.0]),
                         np.array([-0.5, 0.3, 2.0])])
    vel_np_b = np.stack([v0, np.array([2.0, -1.0, 4.0]),
                         np.array([1.0, 2.0, 1.0])])
    pi_vm = jax.jit(jax.vmap(bj.predict_intercept, in_axes=(0, 0, None, None)))
    p_ib, t_ib, f_ib = pi_vm(jnp.asarray(pos_np_b), jnp.asarray(vel_np_b), 1.2, k)
    for s in range(3):
        p_np, t_np = predict_intercept(pos_np_b[s], vel_np_b[s], 1.2, k=k)
        d = max(float(np.max(np.abs(p_np - np.asarray(p_ib[s])))),
                abs(t_np - float(np.asarray(t_ib[s]))))
        assert d < TOL
    assert bool(np.asarray(f_ib).all())
    print("  vmap(predict_intercept) ×3 与 numpy 对齐（<1e-6）")

    # vmap：批量 design_serve_with_drag
    tg_b = jnp.asarray(np.stack([np.array([1.0, 0.05, 0.7]),
                                 np.array([0.9, -0.2, 0.8])]))
    T_b = jnp.asarray(np.array([0.66, 0.70]))
    ds_vm = jax.jit(jax.vmap(bj.design_serve_with_drag,
                             in_axes=(0, 0, None, None, None)))
    p0_b, v_b = ds_vm(tg_b, T_b, k, 1.0, 2.8)
    for s in range(2):
        p0_np, v_np = design_serve_with_drag(np.asarray(tg_b[s]),
                                             float(T_b[s]), k, 1.0, 2.8)
        d = max(float(np.max(np.abs(p0_np - np.asarray(p0_b[s])))),
                float(np.max(np.abs(v_np - np.asarray(v_b[s])))))
        assert d < TOL
    print("  vmap(design_serve_with_drag) ×2 与 numpy 对齐（<1e-6）")

    # jit + 批量：sample_serve_batch（显式 batch 维，随机数来自 jax key）
    key = jrandom.PRNGKey(0)
    ks = jrandom.split(key, 4)
    u_r = jrandom.uniform(ks[0], (6,))
    u_th = jrandom.uniform(ks[1], (6,))
    u_z = jrandom.uniform(ks[2], (6,))
    u_T = jrandom.uniform(ks[3], (6,))
    jss = jax.jit(bj.sample_serve_batch,
                  static_argnames=("feas_radius", "flight_lo", "flight_hi",
                                   "serve_height", "serve_back", "design_k",
                                   "dt", "n_steps_max"))
    p0s, vs, Ts, tgs = jss(jnp.asarray(np.array([1.05, 0.1, 0.75])), u_r, u_th,
                           u_z, u_T, 0.06, 0.65, 0.85, 1.0, 2.8, design_k=0.0125)
    assert p0s.shape == (6, 3) and Ts.shape == (6,)
    assert bool(jnp.isfinite(vs).all())
    assert bool(jnp.all((Ts >= 0.63) & (Ts <= 0.87)))  # clip 到 [lo-grid, hi+grid]
    assert float(jnp.max(jnp.abs((Ts - 0.26) / 0.02 - jnp.rint((Ts - 0.26) / 0.02)))) < 1e-9
    print("  jit(sample_serve_batch) ×6 通过（T 落在 0.02 帧网格上）")

    # jit：drag_force
    jdf = jax.jit(bj.drag_force)
    assert jdf(jnp.asarray(np.array([3.0, -1.5, 2.0])), k, 0.0577).shape == (3,)

    # jit + 显式 batch 维：BallObsBuffer 的 update / read_fit
    hist, st_single, _, buf, meta = _drive_buffers()
    k = meta["k"]
    st2 = buf.make_init(batch_shape=(2,))
    # 用录制的确定性输入驱动两条相同 lane（零噪声、永不 drop）
    for s, p in hist:
        st2 = buf.update_with_draws(st2, jnp.asarray(p), s,
                                    jnp.zeros((2, 3)), jnp.ones((2,)))
    pos2, vel2, k2, ready2 = buf.read_fit(st2, 29, k=k, min_points=8,
                                          max_points=16)
    assert pos2.shape == (2, 3)
    assert float(jnp.max(jnp.abs(pos2 - pos2[0]))) == 0.0  # 两条 lane 一致
    jread = jax.jit(lambda stp, step: buf.read_fit(stp, step, k=k,
                                                   min_points=8, max_points=16))
    pos_j, vel_j, _, ready_j = jread(st_single, 29)
    d_p = float(np.max(np.abs(np.asarray(pos_j) - np.asarray(
        _np_read_fit(hist, 29, k, meta["delay"], meta["ctrl_dt"])[0]))))
    assert d_p < TOL
    # update（key 采样版）smoke：形状与首槽写入
    jupd = jax.jit(buf.update)
    st_u = jupd(buf.make_init(), jnp.asarray(np.array([1.0, 2.0, 3.0])), 0,
                jrandom.PRNGKey(1))
    assert int(np.asarray(st_u.hist_step[0])) == 0
    print("  jit(BallObsBuffer.update/read_fit) 通过；批量 lane (2,) 一致")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"[{name}]")
            fn()
            print(f"PASS {name}")
    print("all tests passed")
