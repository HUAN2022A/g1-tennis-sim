"""RL 阶段 1 · T4：右腕残差策略 PPO 训练入口（train/smoke/eval/probe）。

架构（对应 docs/RL_STAGE1.md D7）：
- 单 env + 单 residual_fn（rl/residual.make_residual_fn）。训练/评估共用同一
  jitted rollout，靠 EnvState.res_params 注入区分：train 注入
  ResidualParams(noise_on=1)，eval 注入 noise_on=0（确定性 mean）——noise_on
  是 0-d 动态叶，切换零重编译（权重热更新同理：carry 值变化、结构不变）。
- collect：reset → state._replace(res_params=train_params) → env.rollout →
  host 侧重建 obs_r（由 trace 的 ctx 分量）、forward mean/value（θ_old）、
  noise=delta-mean、logp_old、reward、GAE（done 吸收态 bootstrap=0）。
- 门控步剔除（D6 × logp 重放）：residual_fn 内部把非激活步 δ 掩码为 0，故
  trace.delta 在这些步是 0 而 mean≠0，噪声=delta-mean 是伪值；host 侧用
  gate = active & fit_ready & mode∈{ARM,SWING} 重建门控掩码，仅 gate 步
  参与 loss（importance ratio 只在这些步有意义）。
- old_params 冻结：collect 后一次性用 θ_old/log_std_old 算 mean/logp_old，
  之后 epoch 内 logp_new 用当前参数，logp_old 不随更新变化。
- GAE 逐环境独立、done 处 bootstrap=0；PPO-clip + 状态无关高斯（log_std
  可学习，clamp 到 [LOG_STD_LO, LOG_STD_HI]）；Adam（optax，经 flax 依赖
  传递安装，见 runbook）。
- checkpoint（pickle，本地受信产物）：policy params + log_std + opt state +
  step + seed + rng；--train --resume 恢复。冒烟 --smoke 断言 loss/params 全
  有限 + 打印吞吐（samples/sec 与 1024 env 外推）。

用法：
    uv run python rl/train_ppo.py --smoke                 # 冒烟（loss 无 NaN + 吞吐）
    uv run python rl/train_ppo.py --train --num-envs 256 # 全量训练
    uv run python rl/train_ppo.py --eval --ckpt <path> --n-balls 200
    uv run python rl/train_ppo.py --probe --n-balls 200  # 鲁棒性探针
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from collections import namedtuple
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random as jrandom

from rl import residual as R
from rl import reward as RW
from rl import tennis_env as te

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = REPO_ROOT / "storage/ckpts/rl/residual_ppo.pkl"

CtxHost = namedtuple("CtxHost", "inter_p site_pos t_arr ball_pos ball_vel ready found")


# ---------------------------------------------------------------------------
# 数值小工具（numpy，host 侧）
# ---------------------------------------------------------------------------

def gaussian_logp_np(noise: np.ndarray, log_std: np.ndarray) -> np.ndarray:
    """状态无关高斯 logp（numpy，host 侧；与 rl.residual.gaussian_logp 同式）。

    常量用 float32 与 jax 侧 jnp.log(2π) 对齐（避免 host float64 vs jax float32
    的精度不对称）。
    """
    var = np.exp(2.0 * log_std)
    return -0.5 * np.sum(noise * noise / var + 2.0 * log_std
                         + np.log(np.float32(2.0 * np.pi)), axis=-1)


def gae(rewards: np.ndarray, values: np.ndarray, active: np.ndarray,
        gamma: float, lam: float):
    """逐环境独立 GAE（吸收态：next 步非 active 时 bootstrap=0）。

    Args:
        rewards/values: (B, N)。active: (B, N) bool（本步是否有效）。
    Returns:
        (adv (B, N) float64, returns (B, N) float64)。
    """
    B, N = rewards.shape
    rewards = rewards.astype(np.float64)
    values = values.astype(np.float64)
    adv = np.zeros((B, N), dtype=np.float64)
    last_gae = np.zeros(B, dtype=np.float64)
    for t in reversed(range(N)):
        if t == N - 1:
            next_val = np.zeros(B)
            cont = np.zeros(B)
        else:
            next_val = values[:, t + 1]
            cont = active[:, t + 1].astype(np.float64)   # 下一真实步才有延续
        delta = rewards[:, t] + gamma * next_val * cont - values[:, t]
        last_gae = delta + gamma * lam * cont * last_gae
        adv[:, t] = last_gae
    return adv, adv + values


def rebuild_obs_r(tr_np: dict) -> np.ndarray:
    """从 trace 的 ctx 分量重建 obs_r（(B, N, 12)）。

    trace 字段到 ctx 的映射：inter_p/inter_p, site_pos/site_pos_pre,
    t_arr/t_arr, ball_pos/fit_pos, ball_vel/fit_vel, ready/fit_ready,
    found/fit_found（字段名不对称见 tennis_env 注释）。
    """
    ctx = CtxHost(
        inter_p=jnp.asarray(tr_np["inter_p"], jnp.float32),
        site_pos=jnp.asarray(tr_np["site_pos_pre"], jnp.float32),
        t_arr=jnp.asarray(tr_np["t_arr"], jnp.float32),
        ball_pos=jnp.asarray(tr_np["fit_pos"], jnp.float32),
        ball_vel=jnp.asarray(tr_np["fit_vel"], jnp.float32),
        ready=jnp.asarray(tr_np["fit_ready"]),
        found=jnp.asarray(tr_np["fit_found"]),
    )
    return np.asarray(jax.jit(R.make_obs_r)(ctx))


_APPLY_CACHE = {}


def forward(module, policy_params: dict, obs_r: np.ndarray):
    """residual 模块前向：(mean (B,N,3), value (B,N))，numpy 进出。

    按 module 身份缓存 jax.jit(apply)——避免热循环内每次新建 jit 包装导致 jax
    编译缓存失效（新 lambda 每次身份不同 → 每个训练迭代都重编译，拖慢吞吐）。
    """
    key = id(module)
    if key not in _APPLY_CACHE:
        _APPLY_CACHE[key] = jax.jit(lambda p, o: module.apply(p, o))
    apply_jit = _APPLY_CACHE[key]
    B, N = obs_r.shape[:2]
    flat = obs_r.reshape(-1, R.OBS_R_DIM)
    mean_flat, value_flat = apply_jit(policy_params, jnp.asarray(flat, jnp.float32))
    mean = np.asarray(mean_flat).reshape(B, N, R.ACTION_R_DIM)
    value = np.asarray(value_flat).reshape(B, N)
    return mean, value


def gate_mask(tr_np: dict) -> np.ndarray:
    """重建 residual_fn 的门控掩码（active & ready & mode∈{ARM,SWING}）。"""
    return (tr_np["active"] & tr_np["fit_ready"]
            & ((tr_np["mode"] == te.MODE_ARM) | (tr_np["mode"] == te.MODE_SWING)))


# ---------------------------------------------------------------------------
# 采集 / 评估
# ---------------------------------------------------------------------------

def collect(env, module, train_params, reset_key, gamma: float, lam: float):
    """一次 rollout + host 侧重建 → 扁平化训练样本。

    Returns:
        dict 含 obs_r/act/old_logp/adv/ret（均已按 gate 掩码展平为 (M, ...)）；
        M 可能为 0（本批无 ARM/SWING 步）。
    """
    state, extras = env.reset(reset_key)
    state = state._replace(res_params=train_params)
    final, tr = env.rollout(state)
    tr_np = jax.tree.map(np.asarray, tr)._asdict()
    fin_np = jax.tree.map(np.asarray, final)._asdict()
    extras_np = jax.tree.map(np.asarray, extras)

    obs_r = rebuild_obs_r(tr_np)                        # (B,N,12)
    mean, value = forward(module, train_params.policy, obs_r)
    noise = tr_np["delta"] - mean                       # (B,N,3)
    logp = gaussian_logp_np(noise, np.asarray(train_params.log_std))  # (B,N)
    rewards, _dones = RW.compute_rewards(env, tr_np, fin_np, extras_np)
    adv, ret = gae(rewards, value, tr_np["active"], gamma, lam)
    gate = gate_mask(tr_np)

    m = gate.reshape(-1)
    return {
        "obs_r": obs_r.reshape(-1, R.OBS_R_DIM)[m],
        "act": tr_np["delta"].reshape(-1, R.ACTION_R_DIM)[m],
        "old_logp": logp.reshape(-1)[m],
        "adv": adv.reshape(-1)[m],
        "ret": ret.reshape(-1)[m],
    }


def eval_hit_rate(env, module, policy_params, log_std, n_balls: int, seed: int):
    """注入 eval params（noise_on=0）跑 n_balls，复用 env._reduce_batch/_summarize。"""
    eval_params = R.ResidualParams(policy=policy_params, log_std=log_std,
                                   noise_on=jnp.array(0.0, jnp.float32))
    B = env.num_envs
    n_batches = int(np.ceil(n_balls / B))
    rec = {k: [] for k in ["hit", "outcome", "tau", "min_dist", "T", "k",
                           "launch", "trig_step", "target_dev", "tau_valid"]}
    t0 = time.time()
    for b in range(n_batches):
        key = jrandom.fold_in(jrandom.PRNGKey(seed), b)
        state, extras = env.reset(key)
        state = state._replace(res_params=eval_params)
        final, tr = env.rollout(state)
        env._reduce_batch(final, tr, extras, min(B, n_balls - b * B), rec)
    wall = time.time() - t0
    return env._summarize(rec, n_balls, wall, verbose=True)


# ---------------------------------------------------------------------------
# PPO 更新（jax）
# ---------------------------------------------------------------------------

def make_update_fn(module, optimizer, clip_eps: float, vf_coef: float,
                   ent_coef: float):
    def loss_fn(params_logstd, batch):
        policy_params, log_std = params_logstd
        obs_r, act, old_logp, adv, ret = batch
        mean, value = module.apply(policy_params, obs_r)
        noise = act - mean
        logp = R.gaussian_logp(noise, log_std)
        ratio = jnp.exp(jnp.clip(logp - old_logp, -20.0, 20.0))
        adv_n = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)
        pg = -jnp.minimum(ratio * adv_n,
                          jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_n)
        vf = 0.5 * (value - ret) ** 2
        ent = R.gaussian_entropy(log_std)
        return jnp.mean(pg + vf_coef * vf - ent_coef * ent)

    @jax.jit
    def update(params_logstd, opt_state, batch):
        (loss, grads) = jax.value_and_grad(loss_fn)(params_logstd, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params_logstd)
        policy_params, log_std = optax.apply_updates(params_logstd, updates)
        log_std = jnp.clip(log_std, R.LOG_STD_LO, R.LOG_STD_HI)
        return (policy_params, log_std), opt_state, loss

    return update


def _tree_finite(pytree) -> bool:
    return all(bool(np.all(np.isfinite(np.asarray(x)))) for x in
               jax.tree_util.tree_leaves(pytree))


# ---------------------------------------------------------------------------
# checkpoint（pickle：本地受信产物）
# ---------------------------------------------------------------------------

def save_ckpt(path, ckpt: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)


def load_ckpt(path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# 鲁棒性探针（固定 δ 摊派下冻结跟踪器非腕关节跟踪误差）
# ---------------------------------------------------------------------------

# 非腕关节 = 除右腕(26:29)外的 26 关节；obs 关节块 6:35（qpos）与 93:122（ref）
# 跟踪误差 = |joint_qpos - ref_qpos|（不是 |joint_qpos - nominal|，后者是"偏离站姿"）
NONWRIST_QPOS = (6, 32)
NONWRIST_REF = (93, 119)
NONWRIST_QVEL = (35, 61)
# 最可能经 last_action 反馈被腕部扰动耦合的子集：右肩(22,23,24)+右肘(25)
COUPLED_JOINTS = (22, 23, 24, 25)


def const_delta_from_params(obs, ctx, params, gating: bool = True):
    """探针固定 δ 残差函数：δ 从 params（(3,) 数组）读取，门控同 residual_fn。

    单函数 + params 注入（state._replace(res_params=δ_vector)）→ 不同 δ 零重编译
    （params 是动态叶，图结构恒定；每 δ 换闭包会导致每次重编译大图，CPU 上极慢）。
    恒值 δ 不耗 rng → 同 seed 逐球配对成立。
    """
    dv = jnp.asarray(params, jnp.float32)               # (3,)
    d = jnp.broadcast_to(dv, (obs.shape[0], R.ACTION_R_DIM))
    if gating:
        a = (ctx.ready & ((ctx.mode == te.MODE_ARM) | (ctx.mode == te.MODE_SWING))
             & ctx.active)
        d = d * a.astype(jnp.float32)[..., None]
    return d


def probe_metrics(tr_np: dict):
    """从 trace 提取非腕跟踪误差（逐关节 max/mean + 耦合子集）+ qvel RMS + 根高度。

    只统计 gate 步（active & mode∈{ARM,SWING}），避免 IDLE/TRACK 稀释。
    """
    gate = (tr_np["active"]
            & ((tr_np["mode"] == te.MODE_ARM) | (tr_np["mode"] == te.MODE_SWING)))
    obs = tr_np["obs"]                       # (B,N,136)
    track = np.abs(obs[..., NONWRIST_QPOS[0]:NONWRIST_QPOS[1]]
                   - obs[..., NONWRIST_REF[0]:NONWRIST_REF[1]])  # (B,N,26)
    qvel = np.abs(obs[..., NONWRIST_QVEL[0]:NONWRIST_QVEL[1]])   # (B,N,26)
    g = gate[..., None]
    track_g = np.where(g, track, 0.0)
    qvel_g = np.where(g, qvel, 0.0)
    n_g = np.maximum(gate.sum(axis=-1, keepdims=True), 1)  # (B,1)

    per_joint_mean = track_g.sum(axis=-2) / n_g            # (B,26)
    coupled = per_joint_mean[:, COUPLED_JOINTS].mean()     # 标量
    return {
        "track_mean": float(per_joint_mean.mean()),
        "track_max_joint": float(per_joint_mean.max(axis=-1).mean()),  # 逐环境最大关节再均值
        "track_coupled": float(coupled),
        "qvel_rms": float(np.sqrt((qvel_g ** 2).sum() / max(gate.sum(), 1))),
        "root_z_min": float(tr_np["root_pos"][..., 2][gate].min()) if gate.any() else float("nan"),
    }


def run_probe(env, delta_sweep=(0.0, 0.1, 0.2, 0.3, 0.5), n_balls: int = 200,
              seed: int = 0, verbose: bool = True):
    """鲁棒性探针：同 seed 逐球配对，扫固定 δ（腕 pitch 轴），报告非腕跟踪误差。

    固定 δ 经 const_delta_from_params + params 注入（只换一次闭包 → 只重编译一次），
    不经 env.evaluate()（它不注入 params）。恒值 δ 不耗 rng → 同 seed 逐球配对。
    """
    B = env.num_envs
    n_batches = int(np.ceil(n_balls / B))
    saved_fn = env.residual_fn
    saved_rollout = env._rollout_jit
    env.residual_fn = const_delta_from_params        # 换一次闭包 → 重编译一次
    env._rollout_jit = jax.jit(env._rollout)
    results = []
    try:
        for dv in delta_sweep:
            delta_val = jnp.asarray([0.0, dv, 0.0], jnp.float32)   # 只扰 pitch 轴
            rec = {"track_mean": [], "track_max_joint": [], "track_coupled": [],
                   "qvel_rms": [], "root_z_min": []}
            for b in range(n_batches):
                key = jrandom.fold_in(jrandom.PRNGKey(seed), b)
                state, _extras = env.reset(key)
                state = state._replace(res_params=delta_val)       # 注入 δ（动态，零重编译）
                _final, tr = env.rollout(state)
                tr_np = jax.tree.map(np.asarray, tr)._asdict()
                m = probe_metrics(tr_np)
                for k in rec:
                    rec[k].append(m[k])
            row = {k: float(np.nanmean(v)) for k, v in rec.items()}
            row["delta"] = float(dv)
            results.append(row)
            if verbose:
                print(f"  δ_pitch={dv:+.2f}: track_mean {row['track_mean']:.4f} "
                      f"max_joint {row['track_max_joint']:.4f} "
                      f"coupled {row['track_coupled']:.4f} "
                      f"qvel_rms {row['qvel_rms']:.4f} "
                      f"root_z_min {row['root_z_min']:.3f}")
    finally:
        env.residual_fn = saved_fn
        env._rollout_jit = saved_rollout
    base = results[0]["track_mean"]
    if verbose:
        print(f"  基线（δ=0）非腕跟踪均值 {base:.4f}；2×阈值 = {2 * base:.4f}")
    return results


# ---------------------------------------------------------------------------
# 冒烟 / 训练 / 主入口
# ---------------------------------------------------------------------------

def run_smoke(num_envs: int = 8, iters: int = 3, seed: int = 0, verbose: bool = True):
    """T4 冒烟验收：loss 无 NaN + 吞吐。返回 dict（含 loss 序列与吞吐）。"""
    key = jrandom.PRNGKey(seed)
    module = R.ResidualPolicy()
    env = te.TennisEnv(num_envs=num_envs, residual_fn=R.make_residual_fn(module))
    key, pk = jrandom.split(key)
    params = R.init_params(pk)
    train_params = params._replace(noise_on=jnp.array(1.0, jnp.float32))

    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
    opt_state = optimizer.init((params.policy, params.log_std))
    update = make_update_fn(module, optimizer, 0.2, 0.5, 0.01)

    losses = []
    t0 = time.time()
    total_samples = 0
    for it in range(iters):
        key, rk = jrandom.split(key)
        batch = collect(env, module, train_params, rk, gamma=0.99, lam=0.95)
        M = len(batch["obs_r"])
        total_samples += M
        if M == 0:
            if verbose:
                print(f"  [iter {it}] 无 ARM/SWING 步，跳过更新")
            continue
        # minibatch 全部（smoke 小规模），或按 minibatch 切
        minibatch = 2048
        mb = min(minibatch, M)
        idx = np.asarray(jax.random.permutation(jax.random.fold_in(rk, it),
                                                M))[:mb]
        b = {k: jnp.asarray(v[idx], jnp.float32) for k, v in batch.items()}
        b["adv"] = b["adv"].astype(jnp.float32)
        b["ret"] = b["ret"].astype(jnp.float32)
        (policy_new, log_std_new), opt_state, loss = update(
            (params.policy, params.log_std), opt_state,
            (b["obs_r"], b["act"], b["old_logp"], b["adv"], b["ret"]))
        params = params._replace(policy=policy_new, log_std=log_std_new)
        train_params = params._replace(noise_on=jnp.array(1.0, jnp.float32))
        loss = float(loss)
        losses.append(loss)
        assert np.isfinite(loss), f"loss NaN/Inf at iter {it}: {loss}"
        assert _tree_finite(params.policy) and _tree_finite(params.log_std), \
            f"params 非有限 at iter {it}"
        if verbose:
            print(f"  [iter {it}] loss {loss:.4f} (M={M}, mb={mb})")

    wall = time.time() - t0
    sps = total_samples / max(wall, 1e-9)
    # 1024 env 外推：T3 实测 CPU 14-18µs/env/控制步 → 1024 env × n_steps 步
    est = 16e-6 * 1024 * env.n_steps
    out = {"losses": losses, "samples_per_sec": sps,
           "wall_s": wall, "total_samples": total_samples,
           "est_1024_batch_s": float(est)}
    if verbose:
        print(f"  smoke: {iters} iters × {num_envs} env，总样本 {total_samples}，"
              f"吞吐 {sps:.0f} samples/s（wall {wall:.1f}s）；"
              f"1024 env 外推 ≈ {est:.2f}s/回合批")
    return out


def run_train(args):
    key = jrandom.PRNGKey(args.seed)
    module = R.ResidualPolicy(hidden=args.hidden, delta_scale=args.delta_scale)
    env = te.TennisEnv(num_envs=args.num_envs,
                       residual_fn=R.make_residual_fn(module))
    key, pk = jrandom.split(key)
    params = R.init_params(pk, hidden=args.hidden, delta_scale=args.delta_scale)

    optimizer = optax.chain(optax.clip_by_global_norm(args.max_grad_norm),
                            optax.adam(args.lr))
    opt_state = optimizer.init((params.policy, params.log_std))
    step = 0
    if args.resume and Path(args.resume).exists():
        ck = load_ckpt(args.resume)
        params = ck["params"]
        opt_state = ck["opt_state"]
        step = ck["step"]
        key = ck["rng"]
        print(f"[train] resume from {args.resume} @ step {step}")
    update = make_update_fn(module, optimizer, args.clip_eps, args.vf_coef,
                            args.ent_coef)

    for it in range(args.total_iters):
        train_params = params._replace(noise_on=jnp.array(1.0, jnp.float32))
        key, rk = jrandom.split(key)
        batch = collect(env, module, train_params, rk, args.gamma, args.gae_lambda)
        M = len(batch["obs_r"])
        if M == 0:
            print(f"[iter {it}] 无 ARM/SWING 步，跳过")
            continue
        for epoch in range(args.epochs):
            idx = np.asarray(jrandom.permutation(jrandom.fold_in(rk, epoch), M))
            for s in range(0, M, args.minibatch):
                sl = idx[s:s + args.minibatch]
                b = {k: jnp.asarray(v[sl], jnp.float32) for k, v in batch.items()}
                (policy_new, log_std_new), opt_state, loss = update(
                    (params.policy, params.log_std), opt_state,
                    (b["obs_r"], b["act"], b["old_logp"], b["adv"], b["ret"]))
                params = params._replace(policy=policy_new, log_std=log_std_new)
                step += 1
        if it % args.log_interval == 0:
            print(f"[iter {it}] loss {float(loss):.4f} M={M} step={step}")
        if args.eval_interval > 0 and (it + 1) % args.eval_interval == 0:
            eval_hit_rate(env, module, params.policy, params.log_std,
                          args.eval_balls, args.seed + 1000 + it)
        if args.ckpt_interval > 0 and (it + 1) % args.ckpt_interval == 0:
            save_ckpt(args.ckpt, {"params": params, "opt_state": opt_state,
                                  "step": step, "rng": key})
    save_ckpt(args.ckpt, {"params": params, "opt_state": opt_state,
                          "step": step, "rng": key})
    print(f"[train] done @ step {step}, ckpt {args.ckpt}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="冒烟：loss 无 NaN + 吞吐")
    ap.add_argument("--train", action="store_true", help="全量 PPO 训练")
    ap.add_argument("--eval", action="store_true", help="加载 ckpt 评估命中率")
    ap.add_argument("--probe", action="store_true", help="鲁棒性探针（固定 δ 摊派）")
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=R.DEFAULT_HIDDEN)
    ap.add_argument("--delta-scale", type=float, default=R.DEFAULT_DELTA_SCALE)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--total-iters", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--n-balls", type=int, default=200)
    ap.add_argument("--eval-interval", type=int, default=10)
    ap.add_argument("--eval-balls", type=int, default=200)
    ap.add_argument("--log-interval", type=int, default=5)
    ap.add_argument("--ckpt-interval", type=int, default=100)
    ap.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    if args.smoke:
        run_smoke(num_envs=8, iters=3, seed=args.seed)
    elif args.train:
        run_train(args)
    elif args.eval:
        module = R.ResidualPolicy(hidden=args.hidden, delta_scale=args.delta_scale)
        env = te.TennisEnv(num_envs=args.num_envs,
                           residual_fn=R.make_residual_fn(module))
        ck = load_ckpt(args.ckpt)
        eval_hit_rate(env, module, ck["params"].policy, ck["params"].log_std,
                      args.n_balls, args.seed)
    elif args.probe:
        module = R.ResidualPolicy()
        env = te.TennisEnv(num_envs=args.num_envs,
                           residual_fn=R.make_residual_fn(module))
        run_probe(env, n_balls=args.n_balls, seed=args.seed)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
