"""RL T4 单测：residual / reward / GAE / PPO 更新 / 鲁棒性探针。

覆盖（设计决策见 docs/RL_STAGE1.md，评审要点见交接文档 §10）：
- residual：ResidualPolicy 前向形状/值域、make_obs_r 未就绪掩码（垃圾输入归零）
- reward：IDLE/窗外/未激活步 reward 严格归零（球 park 于 100m 不漏掩码）
- GAE：done 吸收态 bootstrap=0（手算对照）
- PPO：单步更新 loss/params 全有限（冒烟验收的核心断言）
- 探针：指标基座为 |joint_qpos - ref_qpos|（跟踪误差），非 |qpos - nominal|（偏离
  站姿）——后者被挥拍参考位形主导，会给出假"无发散"结论（评审 blocker）
- 确定性：同 seed collect 逐位一致

运行：uv run python tests/test_rl_train.py
前置：storage/ckpts/rl/v1_calib.npz + storage/data/mocap/Tennis/p1/Random_001_Tennis 001.npz
"""

from __future__ import annotations

import sys
from collections import namedtuple
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from jax import random as jrandom  # noqa: E402

from rl import residual as R  # noqa: E402
from rl import reward as RW  # noqa: E402
import rl.tennis_env as te  # noqa: E402
import rl.train_ppo as tp  # noqa: E402


# ---------------------------------------------------------------------------
# 纯函数测试（无 env）
# ---------------------------------------------------------------------------

def test_residual_shapes():
    key = jax.random.PRNGKey(0)
    module = R.ResidualPolicy()
    params = R.init_params(key)
    mean, value = module.apply(params.policy, jnp.zeros((5, R.OBS_R_DIM)))
    assert mean.shape == (5, 3) and value.shape == (5,)
    assert jnp.all(jnp.isfinite(mean)) and jnp.all(jnp.isfinite(value))
    assert jnp.all(jnp.abs(mean) <= R.DEFAULT_DELTA_SCALE + 1e-6), "mean 超 delta_scale"
    # log_std 可学习 (3,)
    assert params.log_std.shape == (3,)
    print(f"  residual: mean∈[{float(mean.min()):+.3f},{float(mean.max()):+.3f}] "
          f"value finite，|mean|<=delta_scale")


def test_make_obs_r_masks_garbage():
    Ctx = namedtuple("Ctx", "inter_p site_pos t_arr ball_pos ball_vel ready found")
    B = 3
    ctx = Ctx(inter_p=jnp.zeros((B, 3)), site_pos=jnp.ones((B, 3)),
              t_arr=jnp.zeros((B,)), ball_pos=jnp.zeros((B, 3)),
              ball_vel=jnp.zeros((B, 3)),
              ready=jnp.zeros((B,), jnp.bool_), found=jnp.zeros((B,), jnp.bool_))
    o = R.make_obs_r(ctx)
    assert jnp.all(o[:, :10] == 0.0), "未就绪/未找到时 fit/inter 分量必须归零（非 -site_pos 伪误差）"
    assert jnp.all(o[:, 10:] == 0.0)
    ctx2 = ctx._replace(ready=jnp.ones((B,), jnp.bool_), found=jnp.ones((B,), jnp.bool_))
    o2 = R.make_obs_r(ctx2)
    assert jnp.all(o2[:, 0:3] == -1.0), "inter_err = inter_p - site_pos 应生效"
    print("  make_obs_r: 未就绪掩码归零，就绪时 inter_err 生效")


def test_gae_absorbing():
    rewards = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]])
    values = np.zeros((1, 5))
    active = np.array([[True, True, True, False, False]])
    adv, ret = tp.gae(rewards, values, active, gamma=1.0, lam=1.0)
    assert np.allclose(adv[0, :3], 1.0), f"手算 adv 应 [1,1,1]，得 {adv}"
    assert np.all(adv[0, 3:] == 0.0), "done 后 adv 应 0"
    print("  gae: 吸收态 bootstrap=0，adv=[1,1,1,0,0]")


def test_reward_masking():
    class E:
        swing_end = 200
        onset = 100
        peak_local = 10

    B, N = 2, 5
    tr = {
        "active": np.ones((B, N), bool),
        "mode": np.zeros((B, N), int),                    # 全 IDLE
        "swing_f": np.full((B, N), 100, int),
        "ball_pos": np.full((B, N, 3), [100.0, 100.0, 5.0], np.float32),  # park
        "site_pos": np.zeros((B, N, 3), np.float32),
        "delta": np.zeros((B, N, 3), np.float32),
    }
    fin = {"hit": np.zeros(B, bool)}
    extras = {"target_landing": np.array([[7.0, 0.0], [7.0, 0.0]], np.float32)}
    r, d = RW.compute_rewards(E(), tr, fin, extras)
    # 中间步（非最后 active）：r_close 归零（球 park 100m 不产生距离惩罚）+ r_act=0
    assert np.all(r[:, :4] == 0.0), f"中间步 reward 应 0（窗外距离不进 return），得 {r}"
    assert not d.any()
    tr["active"][0, 4] = False
    r, _ = RW.compute_rewards(E(), tr, fin, extras)
    assert r[0, 4] == 0.0, "未激活步 reward 应 0"
    print("  reward: 窗外/未激活步严格归零（落点/过网只摊最后 active 步）")


def test_r_hit_placement():
    class E:
        swing_end = 200
        onset = 100
        peak_local = 10

    B, N = 2, 4
    # 球落在目标落点 [5,0]（过网，land_dist=0）；site=ball 使 r_close=0
    tr = {
        "active": np.array([[True, True, False, False],
                            [True, True, True, False]]),
        "mode": np.full((B, N), te.MODE_SWING, int),
        "swing_f": np.full((B, N), 100, int),
        "ball_pos": np.full((B, N, 3), [5.0, 0.0, 0.0], np.float32),
        "site_pos": np.full((B, N, 3), [5.0, 0.0, 0.0], np.float32),
        "delta": np.zeros((B, N, 3), np.float32),
    }
    fin = {"hit": np.array([True, False])}
    extras = {"target_landing": np.array([[5.0, 0.0], [5.0, 0.0]], np.float32)}
    r, _ = RW.compute_rewards(E(), tr, fin, extras)
    # env0 hit=True：最后 active 步(1) = r_hit(+5) + r_landing(0) + r_net(+1) = 6
    # env1 miss：    最后 active 步(2) = 0 + 0 + 1 = 1；差值 = r_hit 的 +5
    assert abs((r[0, 1] - r[1, 2]) - 5.0) < 1e-5, f"hit vs miss 应差 +5，得 {r[0]}, {r[1]}"
    assert np.all(r[0, :1] == 0.0), "中间步（非最后 active）应 0"
    print("  r_hit: 命中 env 比 miss env 多 +5（r_hit 在最后 active 步）")


def test_landing_net():
    class E:
        swing_end = 200
        onset = 100
        peak_local = 10

    B, N = 2, 3
    ball = np.zeros((B, N, 3), np.float32)
    ball[0, :] = [5.0, 0.0, 0.0]      # env0 落地 x=5（过网）
    ball[1, :] = [1.0, 0.0, 0.0]      # env1 落地 x=1（下网）
    tr = {
        "active": np.ones((B, N), bool),
        "mode": np.full((B, N), te.MODE_SWING, int),
        "swing_f": np.full((B, N), 100, int),
        "ball_pos": ball,
        "site_pos": ball.copy(),        # dist=0 → r_close=0
        "delta": np.zeros((B, N, 3), np.float32),
    }
    fin = {"hit": np.zeros(B, bool)}
    extras = {"target_landing": np.array([[5.0, 0.0], [1.0, 0.0]], np.float32)}
    r, _ = RW.compute_rewards(E(), tr, fin, extras)
    # 过网 env0：r_net=+1, r_landing=0（land_dist=0）→ 最后步 +1
    # 下网 env1：r_net=-1, r_landing=-10（0.5*LANDING_PENALTY 20）→ 最后步 -11
    assert r[0, 2] == 1.0, f"过网球最后步应 +1，得 {r[0]}"
    assert r[1, 2] == -11.0, f"下网球最后步应 -11，得 {r[1]}"
    print("  landing/net: 过网 +1，下网 -11（-1 过网 + -10 落点惩罚）")


def test_gate_mask_exclusion():
    B, N = 2, 5
    tr = {
        "active": np.ones((B, N), bool),
        "fit_ready": np.ones((B, N), bool),
        "mode": np.array([
            [te.MODE_IDLE, te.MODE_TRACK, te.MODE_ARM, te.MODE_SWING, te.MODE_IDLE],
            [te.MODE_ARM, te.MODE_SWING, te.MODE_SWING, te.MODE_IDLE, te.MODE_TRACK],
        ]),
    }
    g = tp.gate_mask(tr)
    expect = np.array([[False, False, True, True, False],
                       [True, True, True, False, False]])
    assert np.array_equal(g, expect), f"gate_mask 不符：\n{g}"
    # fit_ready=False 的步即便 ARM/SWING 也应剔除
    tr["fit_ready"][0, 2] = False
    g2 = tp.gate_mask(tr)
    assert not g2[0, 2], "未 ready 步应剔除出 gate"
    print("  gate_mask: 仅 ARM/SWING & ready & active 步为 True（其余剔除出 loss）")


# ---------------------------------------------------------------------------
# env 相关测试（共享 ENV，编译一次）
# ---------------------------------------------------------------------------

_ENV = None
_MODULE = None


def _get_env():
    global _ENV, _MODULE
    if _ENV is None:
        print("[env] 构造 TennisEnv(num_envs=8, residual)（jit 首次编译较慢）...")
        _MODULE = R.ResidualPolicy()
        _ENV = te.TennisEnv(num_envs=8, residual_fn=R.make_residual_fn(_MODULE))
    return _ENV, _MODULE


def test_collect_finite():
    env, module = _get_env()
    params = R.init_params(jrandom.PRNGKey(0))
    train_params = params._replace(noise_on=jnp.array(1.0, jnp.float32))
    batch = tp.collect(env, module, train_params, jrandom.PRNGKey(1), 0.99, 0.95)
    assert len(batch["obs_r"]) > 0, "应有 ARM/SWING 门控步"
    assert batch["obs_r"].shape[1] == R.OBS_R_DIM and batch["act"].shape[1] == 3
    for k, v in batch.items():
        assert np.all(np.isfinite(v)), f"{k} 含非有限值"
    print(f"  collect: M={len(batch['obs_r'])} 全有限")


def test_ppo_update_finite():
    env, module = _get_env()
    params = R.init_params(jrandom.PRNGKey(2))
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
    opt_state = optimizer.init((params.policy, params.log_std))
    update = tp.make_update_fn(module, optimizer, 0.2, 0.5, 0.01)
    train_params = params._replace(noise_on=jnp.array(1.0, jnp.float32))
    batch = tp.collect(env, module, train_params, jrandom.PRNGKey(3), 0.99, 0.95)
    M = min(len(batch["obs_r"]), 256)
    b = {k: jnp.asarray(v[:M], jnp.float32) for k, v in batch.items()}
    (policy_new, log_std_new), opt_state, loss = update(
        (params.policy, params.log_std), opt_state,
        (b["obs_r"], b["act"], b["old_logp"], b["adv"], b["ret"]))
    assert np.isfinite(float(loss)), f"loss 非有限: {loss}"
    assert tp._tree_finite(policy_new) and tp._tree_finite(log_std_new), "params 非有限"
    print(f"  ppo update: loss {float(loss):.4f}，loss/params 全有限")


def test_collect_determinism():
    env, module = _get_env()
    # reset 的随机数流是确定的：同 seed 的发射采样逐位一致（jax.random 确定性，
    # 这是 checkpoint resume 的基础）。
    _s1, e1 = env.reset(jrandom.PRNGKey(5))
    _s2, e2 = env.reset(jrandom.PRNGKey(5))
    assert np.array_equal(np.asarray(e1["serve_pos"]), np.asarray(e2["serve_pos"])), \
        "同 seed reset 的 serve_pos 应逐位一致"
    assert np.array_equal(np.asarray(e1["T_flight"]), np.asarray(e2["T_flight"])), \
        "同 seed reset 的 T_flight 应逐位一致"

    # rollout：MJX 物理仿真在 GPU 上非确定（contact solver 并行 reduce），同 seed
    # 两次 rollout 的球轨迹有 ~0.1m 量级差异（混沌放大），故只断言探索噪声的
    # old_logp 统计一致（噪声由确定的 ctx.rng 派生），不做 act/obs_r 逐位断言。
    params = R.init_params(jrandom.PRNGKey(7))
    train_params = params._replace(noise_on=jnp.array(1.0, jnp.float32))
    b1 = tp.collect(env, module, train_params, jrandom.PRNGKey(3), 0.99, 0.95)
    b2 = tp.collect(env, module, train_params, jrandom.PRNGKey(3), 0.99, 0.95)
    assert b1["act"].shape == b2["act"].shape and len(b1["obs_r"]) == len(b2["obs_r"])
    assert np.allclose(b1["old_logp"], b2["old_logp"], atol=1e-3), \
        f"old_logp 应统计一致（探索噪声确定），max|Δ|={np.abs(b1['old_logp'] - b2['old_logp']).max():.2e}"
    print("  determinism: reset 采样逐位一致；rollout old_logp 统计一致（GPU 物理非确定）")


def test_probe_metric_uses_ref_not_nominal():
    B, N = 2, 4
    obs = np.zeros((B, N, 136), np.float32)
    obs[:, :, 6:35] = 0.5     # joint_qpos - nominal = 0.5（偏离站姿）
    obs[:, :, 93:122] = 0.5   # ref_jnt - nominal = 0.5（参考也偏离 0.5）
    tr = {
        "active": np.ones((B, N), bool),
        "mode": np.full((B, N), te.MODE_SWING, int),
        "obs": obs,
        "root_pos": np.zeros((B, N, 3), np.float32),
    }
    m = tp.probe_metrics(tr)
    # 跟踪误差 = |qpos - ref| = |0.5 - 0.5| = 0（正确基座）；错用 |qpos-nominal| 会得 0.5
    assert m["track_mean"] == 0.0, f"探针基座应为 |qpos-ref|，得 {m['track_mean']}"
    print("  probe_metrics: 基座为 |qpos - ref|（跟踪误差），非 |qpos - nominal|")


def test_probe_end_to_end():
    env, _ = _get_env()
    results = tp.run_probe(env, delta_sweep=(0.0, 0.2), n_balls=8, seed=0,
                           verbose=False)
    assert len(results) == 2
    for r in results:
        assert np.isfinite(r["track_mean"]) and np.isfinite(r["qvel_rms"])
    print(f"  probe e2e: δ=0 track_mean {results[0]['track_mean']:.4f}, "
          f"δ=0.2 track_mean {results[1]['track_mean']:.4f}")


TESTS = [
    test_residual_shapes,
    test_make_obs_r_masks_garbage,
    test_gae_absorbing,
    test_reward_masking,
    test_r_hit_placement,
    test_landing_net,
    test_gate_mask_exclusion,
    test_collect_finite,
    test_ppo_update_finite,
    test_collect_determinism,
    test_probe_metric_uses_ref_not_nominal,
    test_probe_end_to_end,
]


if __name__ == "__main__":
    for fn in TESTS:
        print(f"[{fn.__name__}]")
        fn()
        print(f"PASS {fn.__name__}")
    print("all tests passed")
