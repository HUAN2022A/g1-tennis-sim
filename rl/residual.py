"""RL 阶段 1 · T4：右腕残差策略（冻结跟踪策略的腕部修正接口）。

LATENT 论文核心洞察：击球/落点精度的来源是"高层直接修正持拍侧手腕 +
低层对腕部扰动鲁棒"。我们的冻结跟踪器 `pns_wo_priv264` 没有为腕部扰动
预留接口，故阶段 1 的方案是：训练一个**小 MLP 残差策略**，输出右腕 3 关节
（动作索引 26:29 = right_wrist_roll/pitch/yaw）的修正 δ，加在
`nn_action[:, 26:29]` 上、`nn2motor_action` 之前——由 `rl/tennis_env.py`
的 residual hook 落位（`ResidualCtx` / `res_params` carry / `StepTrace` 侧录）。

本模块定义与 env 契约正交的纯组件（设计决策 D1-D6，见 docs/RL_STAGE1.md）：
- D2 ResidualPolicy：obs_r -> (mean (B,3), value (B,1))，共享躯干 actor-critic。
- D3 make_obs_r(ctx)：ResidualCtx -> (B,14) 紧凑残差观测（世界系，未就绪
  分量显式掩码为 0，避免把 read_fit/predict_intercept 的 0 占位当成真实误差）。
- D1 ResidualParams / make_residual_fn：env 的 residual_fn 契约
    `(obs, ctx, params) -> δ (B,3)`；探索噪声由 ctx.rng fold_in 派生，经
    `noise_on`（0-d float32 动态叶）缩放——train=1/eval=0 切换**零重编译**
    （不得用 Python bool 走 `if` 分支：carry 下会被 jax 追踪为 traced 值，
    触发 TracerBoolConversionError）。
- D6 门控：δ 仅在 mode∈{ARM,SWING} 且 ready 且 active 时激活，IDLE/TRACK
    与挥拍耗尽后的尾部零扰动。

注意（fail-loud）：make_residual_fn 是 carry 型 residual_fn，params 必须经
`state._replace(res_params=...)` 注入；若 params is None（例如误用
`env.evaluate(residual_fn=...)` 而该 API 从不注入 params），直接 raise 而非
静默返回零残差（否则会拿到 62% 零残差基线而误以为是训练后命中率）。
"""

from __future__ import annotations

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp

OBS_R_DIM = 14          # make_obs_r 输出维度
ACTION_R_DIM = 3        # 右腕 roll/pitch/yaw
DEFAULT_DELTA_SCALE = 0.5
DEFAULT_LOG_STD = -1.0  # 状态无关高斯 std 初值（exp(-1)≈0.37）
DEFAULT_HIDDEN = 64
LOG_STD_LO, LOG_STD_HI = -5.0, 2.0  # std ∈ [6.7e-3, 7.4]，防止坍缩/发散

# 探索噪声派生 seed（固定常量；Date.now/Math.random 在 jit 环境不可用）
EXPLORE_SEED = 0xC0FFEE

# 门控用的模式常量（对齐 rl/tennis_env.py 的 MODE_ARM/MODE_SWING；本地声明
# 避免 import 重模块 mjx/tennis_env 造成循环依赖）
_MODE_ARM = 2
_MODE_SWING = 3


class ResidualPolicy(nn.Module):
    """obs_r -> (mean (B,3), value (B,1)) 的共享躯干 actor-critic。

    mean 头 tanh×delta_scale（δ 在 nn_action 空间，nn2motor 前）；value 头
    线性无激活（Dense(1) squeeze 到 (B,)，GAE 用）。
    """

    hidden: int = DEFAULT_HIDDEN
    delta_scale: float = DEFAULT_DELTA_SCALE

    @nn.compact
    def __call__(self, obs_r: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = nn.tanh(nn.Dense(self.hidden)(obs_r))
        x = nn.tanh(nn.Dense(self.hidden)(x))
        mean = nn.tanh(nn.Dense(ACTION_R_DIM, name="mean")(x)) * self.delta_scale
        value = nn.Dense(1, name="value")(x)[..., 0]
        return mean, value


def make_obs_r(ctx) -> jax.Array:
    """ResidualCtx（或 host 侧重建等价对象）-> (B, 12) 残差观测（世界系）。

    只依赖 ctx 的字段（duck typing）：inter_p / site_pos / t_arr / ball_pos /
    ball_vel / ready / found。未就绪分量显式掩码为 0——read_fit 在 ready=False
    时返回 fit_pos/fit_vel=0、predict_intercept 在 found=False 时返回
    inter_p/t_arr=0，若不掩码会得到 ~-site_pos 的伪误差污染共享躯干。

    分量：
        0:3   inter_err = (inter_p - site_pos)·[ready&found]
        3:4   t_arr·[found]
        4:7   ball_rel = (ball_pos - site_pos)·[ready]
        7:10  ball_vel·[ready]
        10    ready (0/1)
        11    found (0/1)
        12:14 target_landing (世界系 xy，回合内常量，始终有效，无需掩码)
    """
    ready_f = ctx.ready.astype(jnp.float32)             # (B,)
    found_f = ctx.found.astype(jnp.float32)             # (B,)
    rf = (ready_f * found_f)[..., None]                 # ready & found
    inter_err = (ctx.inter_p - ctx.site_pos) * rf
    t_arr = ctx.t_arr[..., None] * found_f[..., None]
    ball_rel = (ctx.ball_pos - ctx.site_pos) * ready_f[..., None]
    ball_vel = ctx.ball_vel * ready_f[..., None]
    tgt = jnp.broadcast_to(ctx.target_landing.astype(jnp.float32), ready_f.shape + (2,))
    obs_r = jnp.concatenate([
        inter_err, t_arr, ball_rel, ball_vel,
        ready_f[..., None], found_f[..., None], tgt,
    ], axis=-1)
    return obs_r.astype(jnp.float32)


class ResidualParams(NamedTuple):
    """residual_fn 的第三参（经 EnvState.res_params carry，pytree 值语义）。

    policy: ResidualPolicy 参数 dict（flax params）。
    log_std: (3,) 状态无关高斯 log 标准差（可学习，训练时 clamp 到 [LO, HI]）。
    noise_on: () 标量 float32，1.0=train（注入探索噪声）0.0=eval（确定性
        mean）。必须是 0-d jax 数组而非 Python bool/float——后者是 jax 静态
        叶，`if` 会按值分叉导致 train↔eval 切换重编译；0-d 数组是动态叶，
        图结构恒定，切换零重编译。
    """

    policy: dict
    log_std: jax.Array
    noise_on: jax.Array


def init_params(key: jax.Array, hidden: int = DEFAULT_HIDDEN,
                delta_scale: float = DEFAULT_DELTA_SCALE,
                log_std: float = DEFAULT_LOG_STD) -> ResidualParams:
    """初始化残差策略参数（含可学习 log_std）。返回 ResidualParams(noise_on=0)。"""
    module = ResidualPolicy(hidden=hidden, delta_scale=delta_scale)
    policy = module.init(key, jnp.zeros((1, OBS_R_DIM), jnp.float32))
    log_std_arr = jnp.full((ACTION_R_DIM,), log_std, jnp.float32)
    return ResidualParams(policy=policy, log_std=log_std_arr,
                          noise_on=jnp.array(0.0, jnp.float32))


def _exploration_noise(key_batch: jax.Array, shape) -> jax.Array:
    """key (B, 2) -> 高斯噪声 (B,)+shape（fold_in 派生，只读不推进流）。"""
    k = jax.vmap(lambda r: jax.random.fold_in(r, EXPLORE_SEED))(key_batch)
    return jax.vmap(lambda r: jax.random.normal(r, shape))(k)


def make_residual_fn(module: ResidualPolicy, gating: bool = True):
    """返回 env 的 residual_fn 契约 `(obs, ctx, params) -> δ (B, 3)`。

    params 为 ResidualParams（经 EnvState.res_params 注入）；params is None
    时 raise（fail-loud，见模块 docstring）。δ = mean + noise_on·std·ε，
    noise_on 为动态标量（train=1/eval=0，零重编译）。gating=True 时 δ 仅对
    mode∈{ARM,SWING} 且 ready 且 active 的步生效，其余置 0。
    """

    def fn(obs: jax.Array, ctx, params: ResidualParams) -> jax.Array:
        B = obs.shape[0]
        if params is None:
            raise ValueError(
                "carry 型 residual_fn 需要经 state._replace(res_params=...) 注入 "
                "ResidualParams；不要用 env.evaluate(residual_fn=...) 直接传本函数"
                "（该 API 从不注入 params，会静默拿到零残差基线）")
        obs_r = make_obs_r(ctx)
        mean, _value = module.apply(params.policy, obs_r)      # (B,3)
        eps = _exploration_noise(ctx.rng, (ACTION_R_DIM,))     # (B,3)
        delta = mean + params.noise_on * jnp.exp(params.log_std) * eps
        if gating:
            active = (ctx.ready
                      & ((ctx.mode == _MODE_ARM) | (ctx.mode == _MODE_SWING))
                      & ctx.active)
            delta = delta * active.astype(jnp.float32)[..., None]
        return delta.astype(jnp.float32)

    return fn


def gaussian_logp(noise: jax.Array, log_std: jax.Array) -> jax.Array:
    """状态无关高斯 log 概率（对最后一个轴求和，即逐样本 logp）。

    noise: (..., D)；log_std: (D,)。返回 (...,)。
    """
    var = jnp.exp(2.0 * log_std)
    return -0.5 * jnp.sum(noise * noise / var + 2.0 * log_std
                          + jnp.log(2.0 * jnp.pi), axis=-1)


def gaussian_entropy(log_std: jax.Array) -> jax.Array:
    """状态无关高斯的熵：H = 0.5·D·log(2πe) + Σ_i log_std_i（标量）。"""
    D = log_std.shape[-1]
    return 0.5 * D * jnp.log(2.0 * jnp.pi * jnp.e) + jnp.sum(log_std)


def _smoke():
    """残差策略前向 + obs_r 构造冒烟（形状 / 有限性 / 掩码 / 门控 / fail-loud）。"""
    key = jax.random.PRNGKey(0)
    module = ResidualPolicy()
    params = init_params(key)

    from collections import namedtuple
    Ctx = namedtuple("Ctx", "inter_p site_pos t_arr ball_pos ball_vel "
                            "ready found mode active rng target_landing")
    B = 4
    ctx = Ctx(
        inter_p=jnp.zeros((B, 3)), site_pos=jnp.zeros((B, 3)),
        t_arr=jnp.zeros((B,)), ball_pos=jnp.zeros((B, 3)),
        ball_vel=jnp.zeros((B, 3)), ready=jnp.ones((B,), jnp.bool_),
        found=jnp.ones((B,), jnp.bool_), mode=jnp.full((B,), _MODE_SWING, jnp.int32),
        active=jnp.ones((B,), jnp.bool_),
        rng=jax.random.split(key, B),
        target_landing=jnp.zeros((B, 2)),
    )
    obs_r = make_obs_r(ctx)
    assert obs_r.shape == (B, OBS_R_DIM), obs_r.shape
    mean, value = module.apply(params.policy, obs_r)
    assert mean.shape == (B, ACTION_R_DIM) and value.shape == (B,)
    assert jnp.all(jnp.isfinite(mean)) and jnp.all(jnp.isfinite(value))

    fn = make_residual_fn(module)
    obs = jnp.zeros((B, 136), jnp.float32)
    d_eval = fn(obs, ctx, params._replace(noise_on=jnp.array(0.0, jnp.float32)))
    d_train = fn(obs, ctx, params._replace(noise_on=jnp.array(1.0, jnp.float32)))
    assert d_eval.shape == d_train.shape == (B, 3)
    # 门控：非 ARM/SWING 或未 ready 或 inactive 时置 0
    ctx_idle = ctx._replace(mode=jnp.full((B,), 0, jnp.int32))
    ctx_nready = ctx._replace(ready=jnp.zeros((B,), jnp.bool_))
    ctx_inact = ctx._replace(active=jnp.zeros((B,), jnp.bool_))
    for c in (ctx_idle, ctx_nready, ctx_inact):
        assert jnp.all(fn(obs, c, params._replace(
            noise_on=jnp.array(0.0, jnp.float32))) == 0.0)
    # 未就绪掩码：ready/found=False 时 inter_err/t_arr/ball_rel/ball_vel 均为 0
    ctx_nf = ctx._replace(ready=jnp.zeros((B,), jnp.bool_),
                          found=jnp.zeros((B,), jnp.bool_),
                          site_pos=jnp.ones((B, 3)))
    o_nf = make_obs_r(ctx_nf)
    assert jnp.all(o_nf[:, :10] == 0.0) and jnp.all(o_nf[:, 10:] == 0.0)
    # fail-loud
    try:
        fn(obs, ctx, None)
        raise AssertionError("params=None 应 raise")
    except ValueError:
        pass
    lp = gaussian_logp(d_train - d_eval, params.log_std)
    ent = gaussian_entropy(params.log_std)
    assert lp.shape == (B,) and jnp.isfinite(ent)
    print(f"  residual smoke: obs_r{obs_r.shape} mean{mean.shape} value{value.shape}"
          f" | mean∈[{float(mean.min()):+.3f},{float(mean.max()):+.3f}]"
          f" | ent={float(ent):.3f} | logp finite {bool(jnp.all(jnp.isfinite(lp)))}")
    print("residual self-test OK")


if __name__ == "__main__":
    _smoke()
