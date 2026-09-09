# RL Stage-1 设计文档与 runbook（右腕残差策略）

更新日期：2026-09-08（T4）
状态：T1/T2/T3 已提交；T4（残差 + reward + PPO + 冒烟 + runbook）本文件落地。
上游：`docs/HANDOFF_2026-09-08.md` §10（阶段 1 计划与验收口径）。

## 0. 目标

冻结跟踪策略 `pns_wo_priv264`（136→2048→1024→512→29 + SiLU）在 MJX 批量网球
环境跑 V1 状态机，零残差命中 **62%**（σ5mm，200 球）。阶段 1 训练一个小 MLP
**右腕残差策略**，输出右腕 3 关节（动作索引 26:29 = right_wrist_roll/pitch/yaw）
的修正 δ，加在 `nn_action[:, 26:29]` 上、`nn2motor_action` 之前，提升击球命中率。

核心风险（LATENT 论文）：我们的冻结跟踪器**没有**为腕部扰动预留接口（LATENT
训练时就移除右腕信号并施加扰动），故加残差可能带崩非腕关节跟踪。**鲁棒性探针**
是该风险的前置信号（见 §4.2）。

## 1. 设计决策（D1-D8）

### D1 残差接口（由 `rl/tennis_env.py` 固定）

- `residual_fn(obs (B,136), ctx ResidualCtx, res_params) -> δ (B,3)`，
  加到 `nn_action[:, 26:29]` 上、`nn2motor_action` 之前。
- `res_params` 经 `EnvState.res_params` carry（pytree 值语义，权重热更新零重编译；
  默认 None）。
- `ResidualCtx` 字段：`ball_pos/ball_vel (B,3), ready (B,), inter_p (B,3), t_arr (B,),
  found (B,), site_pos (B,3), root_pos (B,3), root_yaw (B,), ep_step, launch_step,
  mode, active (B,), rng (B,2)`。`active` 为 T4 hook 增量（挥拍耗尽/NO_SWING 后
  False，供门控抑制尾部无梯度信号的扰动）。
- `StepTrace` 新增侧录（纯记录，零语义改动）：`obs, site_pos_pre, root_pos,
  root_yaw, fit_pos, fit_vel, fit_ready, inter_p, t_arr, fit_found, delta, rng`——
  host 侧重放 obs_r / 探索噪声 / 非腕跟踪误差所需。

**注意（fail-loud）**：`make_residual_fn` 是 carry 型 residual_fn，必须经
`state._replace(res_params=...)` 注入。`env.evaluate(residual_fn=...)` 从不注入
params，会静默拿到零残差基线——故 `make_residual_fn` 在 params is None 时直接
raise，训练/评估一律走"手动 reset→inject→rollout"。

### D2 残差观测 obs_r（`make_obs_r(ctx) -> (B,12)`）

世界系 12 维，未就绪分量显式掩码为 0：

| 段 | 分量 | 掩码 |
|---|---|---|
| 0:3 | `inter_err = inter_p - site_pos` | ready & found |
| 3:4 | `t_arr` | found |
| 4:7 | `ball_rel = ball_pos - site_pos` | ready |
| 7:10 | `ball_vel` | ready |
| 10 | `ready` | — |
| 11 | `found` | — |

不含绝对位姿/root_yaw（场景几何固定，onset 附近近静止）。掩码理由：read_fit 在
ready=False 返回 fit_pos/vel=0、predict_intercept 在 found=False 返回 inter_p/t_arr=0，
不掩码会得到 ~-site_pos 的伪误差污染共享躯干。

### D3 残差策略（`rl/residual.py::ResidualPolicy`）

共享躯干 actor-critic：`obs_r(12) → Dense(64) tanh → Dense(64) tanh →
{mean: Dense(3) tanh×delta_scale(0.5), value: Dense(1) 线性}`。
状态无关高斯 `log_std` 可学习 `(3,)`，clamp ∈ [-5, 2]（std ∈ [6.7e-3, 7.4]），
初值 -1.0（std≈0.37，与 delta_scale=0.5 的动作范围匹配）。

### D4 reward（`rl/reward.py::compute_rewards`，host 侧逐步）

| 项 | 公式 | 权重 |
|---|---|---|
| r_close | 峰值窗（\|swing_f-onset-peak_local\|≤10）内 `-∥ball-site∥` | 1.0 |
| r_hit | 终局命中，摊在每 env 最后一个 active 步（用 `final.hit` 一次性） | +5.0 |
| r_act | `-∥δ∥²` 动作正则 | 0.1 |

**吸收态掩码**：env 是定长 scan + done 吸收态，done 后仍记录。所有 reward 仅
active=True 步非零；窗外/未激活步绝不计算距离（球 park 于 (100,100,5) 时
∥ball-site∥~100m，漏掩码会让 value head 爆炸）。done = ~active 供 GAE 边界 bootstrap=0。

### D5 PPO（`rl/train_ppo.py`）

PPO-clip + GAE + 状态无关高斯：

| 超参 | 值 |
|---|---|
| gamma / gae_lambda | 0.99 / 0.95 |
| clip_eps / vf_coef / ent_coef | 0.2 / 0.5 / 0.01 |
| lr (Adam) / max_grad_norm | 3e-4 / 0.5 |
| epochs / minibatch | 3 / 2048 |
| log_std init / clamp | -1.0 / [-5, 2] |

关键正确性点（评审吸收）：
- **old_params 冻结**：collect 后立即用 θ_old/log_std_old 一次性重放 mean/logp_old，
  冻结后进 epoch；logp_new 每 epoch 用当前参数。
- **门控步 logp 剔除**：residual_fn 内部把非激活步 δ 掩为 0，故这些步
  noise=delta-mean 是伪值；host 侧用 `gate = active & fit_ready &
  mode∈{ARM,SWING}` 重建门控掩码，仅 gate 步参与 loss。
- **GAE 吸收态**：逐环境独立，done/非 active 步 bootstrap=0；advantage 在 loss 内
  标准化。optax 经 flax 依赖传递安装（flax==0.12 → optax 0.2.x）。

### D6 门控（`make_residual_fn(gating=True)`）

δ 仅在 `ready & mode∈{ARM,SWING} & active` 时生效，IDLE/TRACK 与挥拍耗尽后尾部
零扰动。`noise_on` 为 0-d float32 动态叶（train=1/eval=0），探索噪声由 `ctx.rng`
fold_in 派生（只读不推进观测噪声流，host 可重放）——train/eval 切换零重编译。

### D7 训练/评估循环

单 env + 单 residual_fn。collect（`noise_on=1` 注入）→ host 重建 obs_r/mean/
noise=delta-mean/logp_old → reward → GAE → PPO 更新（minibatch shuffle 用
jax.random）；eval（`noise_on=0` 注入）复用 `env._reduce_batch/_summarize` 报命中率。
权重热更新零重编译；train/eval 首次切换各缓存一次 trace（`noise_on` 是动态叶，
非 per-iteration 重编译）。所有随机源走 jax.random（seed 可复现，禁用 time/np.random）。

### D8 文件清单与 runbook

产物：`rl/residual.py`、`rl/reward.py`、`rl/train_ppo.py`、`tests/test_rl_train.py`、
本文件。`rl/tennis_env.py` 增加 `ResidualCtx.active`（T4 hook 增量）。

## 2. 验收标准

### 2.1 冒烟（`--smoke`）

- 每 PPO update 后断言 loss/params 全有限，任一 NaN 即 `sys.exit(1)`（已内建 assert）。
- 打印吞吐（samples/sec）与 1024 env 外推。本机实测：3 iters × 8 env loss
  {21.59, 20.07, 20.73} 全有限；CPU 墙钟含一次性 jit 编译（~60s），编译摊销后
  单 rollout 8 env × ~120 步 ≈ 15ms；1024 env 外推（T3 实测 14-18µs/env/控制步）
  ≈ **1.97s/回合批**（2k-6k 球/分钟，与 handoff T3 一致）。

### 2.2 鲁棒性探针（`--probe`）

同 seed 逐球配对，扫固定 δ（腕 pitch 轴），报告非腕跟踪误差。指标：
- `track_mean` / `track_max_joint`：非腕 26 关节跟踪误差 `|joint_qpos - ref_qpos|`
  的均值 / 逐环境最大关节再均值（**不是** |qpos-nominal|，后者是"偏离站姿"、被
  挥拍参考位形主导，会给出假"无发散"结论）。
- `track_coupled`：右肩(22,23,24)+右肘(25) 耦合子集均值（最可能经 last_action
  反馈被腕部扰动耦合）。
- `qvel_rms`：非腕 26 关节 qvel RMS（振荡探测）；`root_z_min`：根高度最小（失衡信号）。

判据：残差策略实际 δ 幅度下非腕跟踪误差不显著发散（不 >2× 基线）。go/no-go 的
最终门 = 训练后残差策略在配对发球上的非腕误差 + 命中率提升联合评估（固定 δ sweep
只是廉价预检，见交接文档 §10）。

**本机初步信号**（`test_probe_end_to_end`，8 球，δ∈{0, 0.2}）：δ=0 非腕跟踪均值
0.1347 → δ=0.2 为 0.1343（基本持平、略降），固定腕部残差下冻结跟踪器未发散。
完整 5 档 sweep（δ 至 0.5）在服务器 GPU 上跑（见 §4 第 9 条，CPU 上太慢）。

### 2.3 全量回归

```bash
uv run python tests/test_rl_ball.py \
  && uv run python tests/test_rl_env.py \
  && uv run python tests/test_rl_policy.py \
  && uv run python tests/test_rl_train.py
```

本机实测（Windows 需 `PYTHONIOENCODING=utf-8`）：4 文件全绿——test_rl_ball（T2 对齐
1e-15~1e-16）、test_rl_env（T3 5 测）、test_rl_policy（T1 随机批 8.1e-6 / 真实轨迹
8.2e-6）、test_rl_train（T4 11 测，含探针端到端与 collect 确定性）。

## 3. 服务器 runbook（2× RTX 6000 Ada 48G，GPU 1 空闲）

```bash
# 1. 环境（需 Python 3.12）
git clone <repo> && cd <repo>
uv venv --python 3.12 && uv sync --extra cuda   # cuda extra = jax-cuda12-plugin + onnxruntime-gpu

# 2. 数据（gitignored，需手动搬）：4 个 mocap npz
#    本机来源：LATENT 仓库 storage/data/mocap/Tennis/p1/ 或官方 Drive motion_data.zip
scp -r "storage/data/mocap/Tennis/p1/Random_00{1,2,3,4}_Tennis 001.npz" server:repo/storage/data/mocap/Tennis/p1/
#    ⚠️ 阶段 1 仅需 Random_001（DEFAULT_CLIP）；其余 3 个为多 clip 多样性预留。
#    ⚠️ v1_calib.npz / pns_wo_priv264_flax.npz / pns_wo_priv264.onnx 已随 git 追踪，无需搬。
#    ⚠️ calib 与 Random_001 的 traj_len=1874 绑定（env assert len(ref)==traj_len），
#       帧数不一致的 npz 会崩。

# 3. 冒烟（确认 calib traj_len 匹配 + loss 无 NaN）
#    ⚠️ 必须 JAX_DEFAULT_MATMUL_PRECISION=highest：jax GPU 默认 tf32 低精度 matmul，
#       冻结策略 flax 推理偏离 f64 精确值 ~2e-2（正常应 1e-5），拖累跟踪精度（§4 坑 10）
CUDA_VISIBLE_DEVICES=1 JAX_DEFAULT_MATMUL_PRECISION=highest uv run python rl/train_ppo.py --smoke

# 3b. 全量回归（冻结策略/球观测/环境转移的回归，勿跳过）
CUDA_VISIBLE_DEVICES=1 JAX_DEFAULT_MATMUL_PRECISION=highest uv run python tests/test_rl_ball.py
CUDA_VISIBLE_DEVICES=1 JAX_DEFAULT_MATMUL_PRECISION=highest uv run python tests/test_rl_env.py
CUDA_VISIBLE_DEVICES=1 JAX_DEFAULT_MATMUL_PRECISION=highest uv run python tests/test_rl_policy.py
CUDA_VISIBLE_DEVICES=1 JAX_DEFAULT_MATMUL_PRECISION=highest uv run python tests/test_rl_train.py

# 4. 训练（eval 命中率 ≥65% 持续 = 阶段 2 成立）
CUDA_VISIBLE_DEVICES=1 JAX_DEFAULT_MATMUL_PRECISION=highest uv run python rl/train_ppo.py --train --num-envs 512 \
  --total-iters 2000 --eval-interval 10 --eval-balls 200 --seed 0

# 5. 评估 / 探针
CUDA_VISIBLE_DEVICES=1 JAX_DEFAULT_MATMUL_PRECISION=highest uv run python rl/train_ppo.py --eval --ckpt storage/ckpts/rl/residual_ppo.pkl --n-balls 200
CUDA_VISIBLE_DEVICES=1 JAX_DEFAULT_MATMUL_PRECISION=highest uv run python rl/train_ppo.py --probe --num-envs 64 --n-balls 200
```

断线/OOM 恢复：`--train --resume <ckpt>`（checkpoint 为 pickle，含 params+opt+step+seed+rng）。
改变 `--num-envs` 需另建环境（即重编译）。

## 4. 关键技术坑（防复发）

1. `rl/ball_obs_jax.py` 原 import 时全局开 x64 与 MJX(f32-only) 冲突——已改
   `enable_x64()` 显式调用，环境全程 f32。
2. f32 下 GN 有限差分 eps=1e-6 仅 ~2 ulps、雅可比全噪——`tennis_env.py` 内 dtype
   自适应 eps=1e-3 修复。
3. 本机 jax 0.8 CPU batch≥16 GEMM 偶发死锁——策略前向 `lax.map` 分块(chunk≤4)。
4. ONNX 权重必须从 initializer 提取，文档层宽/参数量不可信（实际 136→2048→1024→512→29+SiLU）。
5. **train/eval 标志不能用 Python bool 走 carry**：jax 会把 bool 追踪为 traced 值，
   `if params.train:` 触发 TracerBoolConversionError；用 0-d float32 `noise_on` 算术门控。
6. **鲁棒性探针基座**：跟踪误差 = |qpos − ref|，不是 |qpos − nominal|。
7. MJX box-box 碰撞在球 park 于 (100,100,5) 时对 inf 距离 cast 溢出（良性 RuntimeWarning，
   接触检测不受影响，T3 已实证 62.5% 与 CPU 基线一致）。
8. **本地 Windows 跑测试需 `PYTHONIOENCODING=utf-8`**：测试文件 print 含 Unicode（如 ↔），
   GBK 控制台会抛 UnicodeEncodeError（服务器 Linux 无此问题）。命令形如
   `PYTHONIOENCODING=utf-8 uv run python tests/test_rl_train.py`。
9. **CPU 上 env 编译/rollout 很慢**：本机 jax CPU 编译 `_reset`~65s + `_rollout`~120s，
   单 rollout（8 env × ~150 步）~3-10s（冻结策略前向 + 弹道积分 read_fit/predict_intercept
   为瓶颈）。完整 200 球探针在 CPU 上 ~20min+ 且随内存压力退化；服务器 GPU 快 ~100×，
   **完整验收（探针/训练）应在服务器跑**，本地只做冒烟 + 单测。
10. **jax GPU 默认 tf32（低精度 matmul）**：RTX Ada 上 jax matmul 默认用 tf32，
    冻结策略 flax 推理偏离 f64 精确值 ~2.17e-2（正常应 1e-5）；onnxruntime 仍 1.67e-5。
    **训练/测试必须 `JAX_DEFAULT_MATMUL_PRECISION=highest`**（实测修复后 flax vs f64
    降到 1.10e-5）。test_rl_policy 的 flax↔onnx 对齐阈值按设备自适应（GPU 1e-4、
    CPU 1e-5），因 GPU cuBLAS 累加顺序与 CPU MKL 不同、f32 累差天然 ~5e-5。
11. **GPU MJX 物理非确定**：同 seed 两次 rollout 球轨迹差 ~0.1-0.2m（contact solver
    并行 reduce + 混沌放大），逐位复现不可能。test_rl_train 的 determinism 只断言
    reset 随机数逐位 + old_logp 统计一致（探索噪声由确定的 ctx.rng 派生）。GPU 固有
    特性，不影响训练正确性（RL 本就是随机探索，checkpoint resume 有微小轨迹差异）。

## 5. 下一步（阶段 2/3 判定）

1. 阶段 2 冒烟（GPU 1）：训练曲线成立 = reward 上升且 eval 命中率 ≥65% 持续。
2. 阶段 3 go/no-go：干净配置 ≥80%。
3. 待办遗留：V2 两处修复（`tennis_v2_clipselect.py:215` 加 `refine_k=False` +
   `reset_to_frame` 补 `ball_state`）→ V2 人工标注。
4. RL 长期：球质量/摩擦 model 级 DR、落点/过网 reward（阶段 4）、多 clip 多样性。
