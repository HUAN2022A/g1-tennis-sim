# RL 阶段 3 实验方案（下一轮）

一句话状态：冻结跟踪器 + 3-DOF 右腕残差的过网率被钉在 50–58%，两套 reward 设计均收敛到同一平台，阶段 3 的任务是先用廉价探针判定"天花板在动作权限还是在 reward 梯度"，再据此决定是否扩动作空间。

---

## 1. 背景与现状

任务链：冻结跟踪策略 `pns_wo_priv264` 驱动全身 → 残差 PPO 在右腕 3 关节（动作索引 26:29 = right_wrist_roll/pitch/yaw，`nn2motor_action` 前注入）上加 δ → MJX/jax 批量 env 定长 scan → reward 在 host 侧 numpy 从 rollout trace 计算。

| 阶段 | reward | 命中率 | 过网率 | 落点距(HIT) | 落点距(过网) | net_z_mean_over |
|---|---|---|---|---|---|---|
| Stage 1 | hit-only | **94.6%**（1000 球） | — | — | — | — |
| Stage 2 old | 二值过网 ±1 + (-10 网崖) | 95.0% | 49.5% | 4.18 m | 3.47 m | — |
| Stage 2 new | 密 clearance + 目标可观测 obs(12→14) | 96.0%（峰值 99.0% @it1560） | 51.5%（峰值 58.0% @it1420） | 4.01 m | 3.45 m | **+1.46 m** |

两次 run 均从零训练 2000 iter × 512 envs，每 10 iter 评估 200 球。

**上一轮预测校验**：阶段 2 新 reward 的预测中心为 53%（区间 46–60%），实测均值 51.5%、峰值 58.0% —— 落在预测区间内。这说明"过网率是可预测的低量"这一认知成立，本方案的所有收益估计继续沿用同样的诚实区间范式。

**评测口径（全阶段恒定，不可混用）**：过网 := `landing_x > NET_X`（`rl/tennis_env.py:893` `over_net = landing[0] > NET_X`），与目标域 x∈[4,10] 无关；命中 := `final.hit`；落点距 := `landing_xy` 与 `target_landing` 的 L2。跨 run 比较必须同 seed、同球数口径（200 球诊断 / 1000 球正式）。

---

## 2. 诊断结论（三条证据）

### (i) clearance 项已饱和，退化为旧二值位

新 reward 的过网项 `r_clear = clear_w·tanh((z_cross − NET_HEIGHT)/CLEAR_SCALE)`（`rl/reward.py:119-121`），常量 `CLEAR_SCALE=0.15`、`clear_w=3.0`。

- 实测 `net_z_mean_over = +1.46 m` ⇒ margin = 1.46 − 0.914 = **0.546 m**
- margin/CLEAR_SCALE = 0.546/0.15 = **3.64** ⇒ **tanh(3.64) = 0.9985**
- 峰值斜率 `clear_w/CLEAR_SCALE = 20/m`；工作点斜率 `20·sech²(3.64) ≈ 0.057/m`，仅为峰值的 **~0.3%**

即所有过网球拿到几乎相同的 `r_clear ≈ +2.996`，策略对"过更高/更低"无梯度。更严重的是未过网球走**常数回退**：`clearance = −3.0·CLEAR_SCALE`（`rl/reward.py:120`）⇒ `r_clear ≡ 3·tanh(−3) = −2.9852`，**严格零梯度** —— 落点 x=2.9（差 0.1 m 过网）与 x=1.0（完全下网）拿到完全相同的分数。两者叠加使 `r_clear` 退化为"旧二值位 × 尺度"，这正是 matched-iter（it100–900）平均 −1.8 pp 的机制解释。

### (ii) 能力天花板：两套 reward 都收敛到 50–58%

二值 ±1 与密 clearance 两种设计、两次独立从零训练，过网率均停在 ~50%（均值 49.5% / 51.5%，峰值 58.0%）。**reward 设计不是瓶颈**——若是，第二套应显著更好。瓶颈在驱动权限：

- 实测**可达落点集**：x ∈ [2.0, 5.34]，y ∈ [−1.75, 1.15]（右腕 3-DOF）
- **目标盒**：x ∈ [4, 10]，y ∈ [−4, 4]（`rl/tennis_env.py:116-117` `LANDING_X_LO/HI=4.0/10.0`、`LANDING_Y_LO/HI=-4.0/4.0`）
- 目标盒面积 48 m²，仅 **8.1%** 落在可达包络内；full-amplitude 恒定腕偏置仅移动落点深度 **≤0.15 m**，`corr(landing, target) ≈ 0`

即腕残差对"瞄准"几乎没有权限，落点分布与目标基本无关。

### (iii) 落点全盒余量 ≤0.6 m

落点距(HIT) 4.01 m，而 target 相对固定落点点簇的**几何下限**约 3.6 m ⇒ 现有策略在该指标上距几何下限**仅 ≤0.6 m 余量**。这与 (ii) 一致：不是策略没学好，而是可达集相对目标盒太小，几何期望距离本身就下不来。

---

## 3. 候选方向

> 通用前置（三方向共用）：服务器 ckpt 同步。本地 `storage/ckpts/rl` 只有 `residual_ppo.pkl` + `v1_calib.npz`；`residual_landing.pkl / residual_clear_aim.pkl` 在服务器 `hgpt-rl` 容器内，所有基于现有 ckpt 的探针/几何校正必须在服务器上跑。

### 方向 A — 扩展残差动作空间：右腕 3 关节之外纳入右肘(25)与右肩俯仰(22)

**假设**：过网率被钉在 ~51–58% 的根因是腕 3-DOF 的可达落点集（x 上界 5.34）远小于目标域（x 上界 10）。把残差扩展到右臂近端 `right_shoulder_pitch(idx22)` 与 `right_elbow(idx25)`，可直接扩大击球点深度/速度的可控范围。物理依据：这两个关节的每-nn-单位增益是腕 pitch/yaw 的 **5.89×**（`ACTION_SCALE = TORQUE_LIMIT/KPs`，`tracking/constants.py:182/205/513`：index22/25/26 = 25/14.2506 = 1.7543 rad，index27/28 = 5/16.7783 = 0.2980 rad；乘 `nn2motor` 的 `action_scale=0.25` 后 **0.4386 vs 0.0745 rad/nn-unit**），且位于臂链根部、杠杆臂更长。

**具体改动**

| 文件 | 位置 | 改法 |
|---|---|---|
| `rl/residual.py` | `:36-41` 常量区 | `ACTION_R_DIM` 3→5；新增单一真源 `RESIDUAL_JOINT_IDX = (22, 25, 26, 27, 28)`（顺序即 δ 通道序）；新增 `DEFAULT_DELTA_SCALE_VEC`，允许 `delta_scale` 为标量或长度 `ACTION_R_DIM` 的序列（逐关节缩放） |
| `rl/residual.py` | `:52-68` `ResidualPolicy.__call__` | `delta_scale` 由 float 改为可广播序列；`mean = jnp.tanh(nn.Dense(ACTION_R_DIM, name="mean")(x)) * jnp.asarray(self.delta_scale)`；Dense 输出维自动跟随 `ACTION_R_DIM`（`:66` 无需改数值） |
| `rl/residual.py` | `:119-127` `init_params` | `delta_scale` 透传向量化；`log_std (ACTION_R_DIM,)` 已泛型 |
| `rl/residual.py` | `:182-237` `_smoke` | 字面量 `(B,3)` 在 **`:211`**（非草稿所称 `:204`，`:204` 已是泛型 `assert mean.shape == (B, ACTION_R_DIM)`）；改 `:211` 断言与 `:234-236` 打印为 `ACTION_R_DIM` |
| `rl/tennis_env.py` | 顶部 import + `:582`（唯一注入点） | 新增 `from rl import residual as RS`（无循环依赖，`residual.py` 本地声明 `_MODE_ARM/_MODE_SWING`）；`:582` `nn_action.at[:, 26:29].add(delta)` → `nn_action.at[:, jnp.asarray(RS.RESIDUAL_JOINT_IDX)].add(delta)`；`:584` 的 `jnp.zeros((B,3))` → `(B, RS.ACTION_R_DIM)`；更新 `:20-24` 与 `:325-331` docstring |
| `rl/train_ppo.py` | `:263-267` `NONWRIST_*` | 连续切片 `NONWRIST_QPOS=(6,32)/NONWRIST_REF=(93,119)/NONWRIST_QVEL=(35,61)` 换成显式排除集 `RESIDUAL_EXCLUDE={22,25,26,27,28}`（joint j 的 obs 列：qpos=6+j, qvel=35+j, ref=93+j）。**否则探针会把残差有意施加的肘/肩偏移误判为"跟踪发散"，产生假阴性** |
| `rl/train_ppo.py` | `:270-283` `const_delta_from_params`、`:313-356` `run_probe` | 保持泛型但 `run_probe` 改为构造长度 `ACTION_R_DIM` 的 δ 向量（只填被测关节槽位，其余 0）；新增逐关节 `j∈{22,23,24,25,26}` × `δ∈{−0.5,−0.25,0,+0.25,+0.5}` 扫描；每次报告 (a) 非残差 track_mean/max、root_z_min、qvel_rms；(b) 配对同 seed 的 Δlanding_x / Δlanding_y 与过网率；(c) 命中率；(d) 残差关节"实到位移 vs 指令位移"（δ·gain）以探测 PD/力矩饱和 |
| `rl/train_ppo.py` | `:469-472` `save_ckpt`、`:476-504` CLI、`:510-521` eval/probe 分支 | ckpt dict 增存 `action_joints` 与 `delta_scale`；eval/probe 分支必须从 ckpt 读取该配置重建 module+env，而非用 CLI 默认值，否则静默形状不匹配或 δ 通道错位；新增 `--action-joints`（默认 `'22,25,26,27,28'`）与 `--delta-scale`（支持逐关节 csv） |
| `tests/test_rl_train.py` | `:48/50/52/245` | `:48` `mean.shape==(5,3)`→`(5, R.ACTION_R_DIM)`；`:50` 容差改读向量化 `delta_scale`；`:52` `log_std.shape==(3,)`→`(R.ACTION_R_DIM,)`；`:245` `batch['act'].shape[1]==3`→`R.ACTION_R_DIM` |
| `rl/reward.py` | `:105` `r_act` | 无需改（`np.sum(delta*delta, axis=-1)` 末维泛型）。**但 reward 侧饱和必须同批修复**（见方向 B），否则新动作权限的学习信号被压制 |

**预期收益（诚实，已按评审下调）**：未做探针前为 **过网率 +0~+8 pp，中心 ~+4 pp（约 55–56%）**，命中率有下行风险；**不是草稿的 65%**。只有当前置探针同时满足放行条件，才上修至乐观区间 **58–64%**。即便上修，收益也只在 ~48% 未过网球里翻转其中一部分。`land_dist(HIT)` 期望基本不变（目标盒仍不可达）。

**GPU 成本**：前置探针 **≤0.5 h**（5 档 δ × 5 关节 × 200 球，无训练；参考现有 200 球探针 GPU 上数分钟量级）。全量重训 **≈15.4 h**（2000 iter × 512 envs，与阶段 1 实测一致）。**诚实修正草稿的自相矛盾**：若真跑"run1 统一 0.5 + run2 逐关节等物理权限"两轮消融，成本为 **~30 h**；建议先只跑 run1（统一 `delta_scale=0.5`，隔离两个新关节的净效应、保持腕子策略与探索噪声语义不变），run2 视 run1 结果再定。**动作空间维度改变后旧 ckpt（mean 头 64→3）连 `--eval` 都无法加载**（不只不能 resume），必须新建环境重建。
**风险**：(a) **首要失败模式**——冻结跟踪器是闭环，本身也跟踪 arm 关节参考轨迹，PD 会把肩/肘扰动拉回；最可能的结局不是发散，而是**落点几乎不变**（这条应作为首要 kill 理由，而非次要风险）。(b) PD/力矩饱和：参考挥拍已接近肩部力矩上限（`TORQUE_LIMIT[22]=25 Nm`），指令 δ 未必被物理实现。(c) reward 饱和压制新权限学习。(d) 探索噪声不均：纳入 6 倍物理增益差的关节后，状态无关 `log_std` 在物理空间严重偏斜（此即引入逐关节 scale 的动因）。(e) ckpt 与配置失配（eval 目前只从 CLI 重建 module，`train_ppo.py:510-516`）。(f) 命中率回归：近端大偏移可能造成挥拍抖动、拍-球时序漂移（tau 离散度上升）。

**证伪判据（廉价探针，同一实验同时做前置验证与证伪，全程不训练）**：冻结跟踪器上对每个候选关节 `j∈{22,25}` 施加固定 `δ∈{−0.5,−0.25,0,+0.25,+0.5}`（同 seed 逐球配对，200 球），测配对 `Δlanding_x` 均值、过网率、非残差 `track_mean`（排除 `{22,25,26,27,28}`）、`root_z_min`、**命中率**。
- **判死**：
  - (a) `max_{j∈{22,25}, δ=±0.5} |mean Δlanding_x| < 0.3 m` —— 新增关节对落点深度无可利用权限，方向不可能突破天花板，直接 kill；
  - (b) `δ=0.25` 时命中率 < **基线 −1 pp**，或非残差 `track_mean > 2×基线`（基线 **0.2713**），或 `root_z_min` 较基线下降 >0.02 m —— 跟踪器对近端扰动不鲁棒，任何重训都是追移动靶，kill（或缩 scale 后再测）。
- **放行**：`max |Δlanding_x| ≥ 1.0 m` 且 (b) 通过。
- **灰区**：0.3–1.0 m 之间需人工判断，可加测 `shoulder_roll(23)`。

### 方向 B — 修复过网奖励梯度（让 `r_clear` 真正带梯度）

**假设**：阶段 2 新 reward 打不过旧的 ±1 位，不是"过网不可学"，而是 `r_clear` 在当前工作点**双重失效**：(1) tanh 深饱和（见 §2(i)，局部斜率仅峰值 0.3%）；(2) 未过网常数回退，梯度严格为 0。放大 `CLEAR_SCALE` 让工作点回到 tanh 线性区，并把常数回退换成连续 signed margin，就能让 ~48% 未过网球获得与"差多远"成比例的梯度。**上限由腕部落点权限（full-amplitude 仅 ~0.15 m）封顶**。

**具体改动**

| 文件 | 位置 | 改法 |
|---|---|---|
| `rl/reward.py` | `:34-38` 常量区 | **B1（唯一必需的一行）**：`CLEAR_SCALE` 0.15 → **0.5**（工作点斜率回到 `6·sech²(1.092) = 2.19/m`，峰值 6/m 的 36%；threshold 处斜率由 20/m 降到 6/m，3.3× 钝化——见风险）。新增 **B2** 常量：`VX_EPS=0.02`（前向步阈值，=1.0 m/s @50Hz，排除来球 −x 与 park x=100/vx=0）、`X_SANE=15.0`、`SHORT_K=1.0`、`FALLBACK_MARGIN=-3.0*CLEAR_SCALE`（=−1.5，使无前向段回退仍 `r=−2.9852`，零回归）。`DEFAULT_WEIGHTS` 不动 |
| `rl/reward.py` | `:53-71` `net_cross_z` 之后 | 新增 `net_margin_z(ball, active, net_x=NET_X)->(B,)`：`vx=np.diff(x,axis=1,prepend=x[:,:1])`；`airborne=~(np.cumsum(z<GROUND_Z,axis=1)>0)`（首落地帧前）；`seg=active & (vx>VX_EPS) & (x<X_SANE) & airborne`；`x_fwd=np.where(seg,x,-inf)`；`has_fwd=seg.any(1)`；`k=argmax(x_fwd,1)`；`margin_nc=z_at-NET_HEIGHT-SHORT_K*maximum(0, net_x-x_max)`；`z_cross,crossed=net_cross_z(...)`；返回 `where(crossed, z_cross-NET_HEIGHT, where(has_fwd, margin_nc, FALLBACK_MARGIN))`。**关键**：`vx>0` 前向门控排除来球（发球点 x≈4.85>net_x=3，来球向下穿越 x=3；朴素"最近 x=NET_X 点"会误取来球弧顶）；两分支在 `x_max→net_x⁻` 处连续 |
| `rl/reward.py` | `:119-121` | 替换为 `margin = net_margin_z(ball, active); r_clear = (clear_w*np.tanh(margin/CLEAR_SCALE)).astype(np.float32)`；删除 `np.where(crossed, ..., -3.0*CLEAR_SCALE)` 常数回退；`:122` 终局摊派不变 |
| `rl/reward.py` | `:7-15` docstring + `:34` 注释 | 更新公式描述与 `CLEAR_SCALE=0.5` 的线性区说明 |
| `tests/test_rl_train.py` | `test_net_clearance_reward`（`:139-168`） | 新增：三个"前向、未落地、未过网"合成球（落点 x = 2.9/2.0/1.0，z_at_max=0.05），断言 `r_clear` 随落地 x 单调递增且两两不等（旧代码三者恒为 −2.9852）；保留 ball[2] x=10（vx=0，无前向段）断言回退仍最负 |

**预期收益（诚实，已按评审修正）**：**过网率 +1~+3 pp，中心 ~+2 pp（51.5% → 53–54%）**，与种子噪声同量级，**很可能不显著**；天花板仍 ~58–60%。可分离置信度：**(a) "r_clear 恢复梯度、未过网球不再零梯度" = high（数学确定，勿误读为过网率增益 high）**；(b) "过网率因此 +1.5~3.5 pp" = medium-low。落点距基本不变。

**GPU 成本**：改动零 GPU、零 jit 重编译（reward 在 host 侧 numpy 的 `collect` 里算，`train_ppo.py:172`，~25 行 O(B×N) numpy，微秒级）。验证：**优先 300 iter × 512 从零 A/B**（省成本）或 `--resume`（注意 resume 带入旧 value 头，reward 终局尺度变了需再校准，只适合看方向）。完整 2000 iter ≈15.4 h。**结论：B 的修复确定正确且应做，但"值得为它单独烧 15.4 h full retrain"不成立——应并入选定方向（A）的重训。**

**风险**：(a) **阈值钝化（最主要，可能吃掉全部收益）**：`CLEAR_SCALE` 0.15→0.5 把 margin=0 处决策边界斜率从 20/m 降到 6/m（3.3×），margin=−0.05 m 的下网惩罚从 −0.965 降到 −0.299——"差一点过网"的推力变弱，可能抵消新梯度。**折中建议 `CLEAR_SCALE=0.4`（斜率 7.5/m，工作点 23% 峰值），或分 0.4/0.5 两档消融**。(b) 吊球钻空：放大 scale 后 `r_clear` 在过网后 ~1.0 m 内仍随高度升，理论激励更高弧线；落点项是唯一对冲，若 clear_w 相对 landing_w 占优可能出现"过网率升、land_dist 恶化"；**用 `land_dist_mean_over` 与 `over_net_rate` 同看**（决策阈值：`land_dist_mean_over` 恶化 >0.3 m 即判 reward hacking）。(c) 回退误判：接触后球短暂 `|vx|≤VX_EPS` 会被判 `has_fwd=False`，但这些是接近死球的低速回击，影响面小。(d) value 头瞬态：过网球终局 `r_clear` 从 +2.996 变 +2.39、未过网变连续；回退值刻意保持 −2.9852 以缩小冲击，仍建议看前 ~100 iter explained-variance。(e) 无 CLI 权重通道，消融需改源码 + SFTP。

**证伪判据（两级）**：
1. **纯函数诊断（~1 分钟，先做）**：构造三个前向、未落地、未过网合成球（落地 x=2.9/2.0/1.0），跑 `compute_rewards`——旧代码三者 `r_clear` 恒为 −2.9852，新代码须严格单调且两两不等；并对 margin=0 求数值斜率，新代码在 margin=0.546 m 处斜率应 >1.5/m（旧 0.055/m）。若仍给常数或斜率不升，则实现接错，回改代码。
2. **训练级**：300 iter × 512 从零跑 B1+B2，matched-iter（it100/200/300）比较 `over_net_rate` 与 `residual_clear_aim.pkl` 历史曲线（it100–900 均值 51.5%）。若新曲线不高于旧曲线 +1 pp、且 collect 出的 batch 里过网球终局 `r_clear` 方差仍≈0（仍饱和），则"恢复梯度能提升过网率"被证伪 → 天花板支配，转向方向 A。同时用 `land_dist_mean_over` 检查吊球钻空（恶化 >0.3 m 判定 reward hacking）。

### 方向 C — 任务/目标域重定义（C1）+ 非作弊难度课程（C2）

**C1 假设（算术命题，可当场证伪）**：缩域带来的落点距下降绝大多数是**几何效应，不是能力提升**。因 `corr(landing,target)≈0`，落点分布与目标无关，`landing_dist` 退化为"固定落点点簇到目标框的期望距离"。

**C2 假设（实证命题，需一次完整训练）**：在全框上做"近网→全框"难度退火，末端分布严格等于原全框均匀分布，不改变任务定义；但在既有"腕部能力天花板"诊断下，大概率带不来实质提升。

**具体改动**

| 文件 | 位置 | 改法 |
|---|---|---|
| `rl/tennis_env.py` | `:116-117` 之后常量区 | 新增 `CURRIC_X_LO/HI = 3.3/5.3`、`CURRIC_Y_LO/HI = -1.5/1.0`（网后、100% 落在实测可达包络内）。注释标注"**此为任务重定义框，非法定评估域**" |
| `rl/tennis_env.py` | `_reset`（`:693`、目标采样 `:722-727`） | 签名改 `def _reset(self, rng, curriculum=1.0)`；保留现有 `split(k_landing,2)` 取 `u_lx/u_ly` 不变（m=1.0 逐位复现全框目标），额外用 `u_mix=uniform(fold_in(k_landing, SALT),())`（**不改 split 支数以免动到既有 RNG 流**，`SALT` 需给定值）；`m=clip(curriculum,0,1)`；`use_full=(u_mix<m)`；`x_lo/x_hi/y_lo/y_hi` 用 `jnp.where(use_full, LANDING_*, CURRIC_*)` 选择；`m` 为 0-d float32 动态叶 → 切换零重编译 |
| `rl/tennis_env.py` | 公开 `reset`（`:780-783`） | `def reset(self, seed=0, curriculum=1.0): ... return self._reset_jit(key, jnp.float32(curriculum))`。默认 1.0 = 现行为，旧调用（`ENV.reset(seed=2)` / `env.reset(key)`）向后兼容 |
| `rl/tennis_env.py` | `_reduce_batch`（`:846-895`，landing 段 `:889-895`） | `rec` 增加键 `landing_y / target_x / target_y`；追加 `rec['landing_y'].append(float(landing[1]))`、`rec['target_x'].append(float(tl[0]))`、`rec['target_y'].append(float(tl[1]))` |
| `rl/tennis_env.py` | `_evaluate_inner`（`:830-832`）/ `_summarize`（`:897-951`） | rec 初始化字典补 `landing_y/target_x/target_y`；`_summarize` 增算 `corr_x = corr(landing_x, target_x)`、`corr_y = corr(landing_y, target_y)`（空/常数时 None）并打印一行"aim corr" |
| **`rl/train_ppo.py`** | **`:192-194` `eval_hit_rate`** | **必改（否则 KeyError）**：该处有独立于 `_reduce_batch` 的 rec 字典，`--eval` 与 eval 循环均走它。必须同步加 `landing_y/target_x/target_y` 三键，否则 C1"零成本几何校正"的 eval 路径直接崩 |
| `rl/reward.py` | `landing_xy`（`:41-50`）之后 | 新增纯 numpy `geometric_land_dist(landing_xy, box_lo, box_hi, n=401)`（规则网格求平均期望距离，确定性、无 RNG）与 `remap_targets(target_xy, src_box, dst_box)`（仿射重映射） |
| `rl/train_ppo.py` | `collect`（`:154-183`，reset `:161`）/ `eval_hit_rate`（`:186-203`，reset `:198`） | 各加 `curriculum: float = 1.0` 参数并透传 `env.reset(..., curriculum=curriculum)`（eval 默认全框，头条口径不变） |
| `rl/train_ppo.py` | `run_train`（`:423-473`）/ `main`（`:476-504`） | 加 `--curriculum {none,near2full}`（默认 none）与 `--curriculum-iters/--curriculum-frac`。训练结束追加两次 eval：`curriculum=1.0`（全框头条）与 `=0.0`（近网诊断） |
| `tests/test_rl_train.py` | `TESTS`（`:321-336`） | 新增 `test_curriculum_targets`（m=1.0 逐位一致 / m=0.0 全落缩域 / 0<m<1 混合）与 `test_geometric_land_dist`（缩域期望 < 全框，remap 后预测≈缩域实测） |

**预期收益（诚实）**：**C1（明确标注 TASK REDEFINITION）**：`land_dist_mean` 从 4.01/3.45 m 降到 ~0.9–1.3 m，看似改善 ~2.5 m；但按同一条落点序列做几何重配对的预测值 `D_geo` 就是 0.9–1.3 m ⇒ **跌幅 ≥90%（1.79–3.12 m，中位 ~2.4 m）是纯几何**；**真实能力增益 `Δ_skill = D_geo − D_obs(缩域) ≈ 0`**。`over_net_rate` 不受目标域影响，预期不变（50–58%）；hit 率不变（95–96%）。**C2（课程）**：末端分布 = 原全框，口径可比；预期全框过网率 **+0~+3 pp（中心 ~0–1 pp，噪声内）**、over-net 落点距 −0~0.3 m，**天花板不变**。综合：C 方向**不产生可证的能力提升，价值限于诊断与证伪**。

**GPU 成本**：C1 几何命题 **0 训练成本**（现有 ckpt 上跑 200 球 eval 拿 landing/target 数组，host 侧 numpy 几秒），代码改动 ~40 行；若要真跑缩域训练，1 次 2000 iter × 512 env。C2 **必须 1 次完整 2000 iter × 512 env 训练 + ~50 行改动**（reset 透传 curriculum 属运行期动态叶，零重编译）——**先 300 iter 验方向，无独立理由不烧 full**。

**风险**：(a) **C1 核心风险：把 ~2.5 m 几何跌幅当成能力提升对外汇报**——必须三值并列 `D_full / D_geo / D_obs(缩域)` 并标注 ≥90% 为几何。(b) 缩域 x∈[3.3,5.3] 是"网后 0.3–2.3 m 小球/放短"区，`target_landing` 在 obs(14 尾 2 维)里近乎常量、观测特征失效。(c) 即便缩到 2.0 m，腕部权限仍仅 ~0.15 m（覆盖 7.5%），若缩域 run `corr` 仍≈0，C1 只是换个小框的同一非瞄准行为。(d) 改 reset 随机流会让 extras/target 与旧 run 不再逐位一致——必须用 `fold_in` 派生 `u_mix` 而非改 `split` 支数。(e) **C2 可能白烧一次完整训练**（既有诊断预示课程解决的是优化难度而非能力上限），且可能 curriculum trap（锁进短回球，全框末端不恢复）。(f) C2 评估口径必须 `curriculum=1.0` 固定全框，否则期末指标被课程分布污染。(g) 两变体都不动 reward，若真正瓶颈在 reward 结构，C 不触及。

**证伪判据**：**C1（零训练成本，可立即执行）**：现有 ckpt 上跑 200 球全框 eval，取逐球落点 `L_i` 与目标 `T_i`，算 (a) `D_full = mean||L_i−T_i||`（应≈3.45 over-net）；(b) `D_geo = mean_i gridmean_{T∈缩域}||L_i−T||`（应落在 0.9–1.3 m）；(c) `D_align = mean||L_i − remap(T_i)||`。判据：若 `|D_align − D_geo|` 很小且 `corr(Lx,Tx)≈corr(Ly,Ty)≈0`，则缩域跌幅为纯几何，C1 不具备能力增益（**这正是预期结果，也是把 C1 标为"重定义"的依据**）；反之若 `D_align` 显著低于 `D_geo`（低 0.3 m 以上）且 corr 明显非零，才说明缩域暴露了被大框掩盖的瞄准能力。**C2（需一次训练）**：相同 512env/2000iter/seed、仅 curriculum 开关做 A/B，在 `curriculum=1.0` 全框 eval 比对。**预先登记接受阈值**：过网率至少 **+5 pp（≥56.5%）**，或 over-net 落点距至少 **−0.3 m（≤3.15 m）**，且 hit 率不低于基线 −1 pp。两项均未达则判定课程无实质增益并弃用。

---

## 4. 推荐顺序与决策树

排序原则：**按"性价比 × 是否击中真瓶颈"排序**。B 最便宜且技术零错，但被腕权限钉死；A 是唯一直接攻击真瓶颈（驱动权限）的方向，但收益未实测、成本最高；C 最诚实但本质不产生改进。**三方向都不改冻结跟踪器本身，"闭环把扰动抵消"这一最可能的失败模式，只有 A 的探针能覆盖。**

```
Step 0  廉价诊断（0 GPU·h，~分钟级；全部在服务器 hgpt-rl 上跑，因 ckpt 在服务器）
  ├─ B 纯函数测试：合成三球，确认 r_clear 单调性 + margin=0 斜率 >1.5/m（1 分钟，本地可跑 pytest）
  └─ C1 几何校正：现有 ckpt 上 200 球全框 eval，算 D_full / D_geo / D_align（~分钟级）
        → 预期 |D_align−D_geo| 小、corr≈0 ⇒ C1 钉死为 TASK REDEFINITION，不再单独投训练

Step 1  A 前置探针（≤0.5 GPU·h）——这是决定后续所有投入的 gate
  ├─ max|Δlanding_x| < 0.3 m            → kill A：动作空间无权限，天花板不在 DOF
  ├─ 0.3–1.0 m（灰区）                   → 人工判断，加测 shoulder_roll(23)
  └─ ≥1.0 m 且鲁棒性通过（hit ≥ 基线−1pp、track_mean ≤ 2×0.2713、root_z_min 降 ≤0.02 m）
        → 放行 Step 2

Step 2  A run1 全量重训（≈15.4 GPU·h），**同批并入方向 B 的 reward 修复**
        （统一 delta_scale=0.5，隔离两个新关节净效应）
  ├─ 过网率 ≤55%（与旧曲线 matched-iter 同量级） → 天花板不在动作空间
  │     → 转向"降低跟踪器参考挥拍在 SWING 段对持拍臂的跟踪刚度"这类更小杠杆（见 §6）
  ├─ 过网率升但 land_dist(HIT) 平坦            → 权限在深度、不在瞄准；天花板在别处（接触时序/球拍交互）
  └─ 过网率与落点距同时改善，且 hit 不降        → 上修乐观区间 58–64%，再考虑 run2（逐关节等物理权限）

Step 3  仅当有独立理由（如需证伪课程假设）才跑 C2，且**先 300 iter**；接受阈值预登记（+5pp 或 −0.3 m）
```

关键 gate 逻辑：**先跑廉价探针；若去饱和后的 clearance 项仍把过网率压在 ≤55%，则扩动作空间；若扩了动作空间落点仍平坦，则天花板在别处（tracker 参考挥拍 / 接触时序），不在 DOF。** 任何一次 full retrain 前必须过对应 gate，避免为噪声烧 15 h。

---

## 5. 复现命令

**同步（GitHub 服务器不通，代码走 SFTP）**：本地改动完成后，SFTP 同步到服务器 `~/tennis-rl`（挂载到容器 `/workspace/Humanoid-GPT`）。SSH/SFTP：`11e1c7a362341f71.natapp.cc:6382`，user `quzh`。

**训练必须带 `JAX_DEFAULT_MATMUL_PRECISION=highest`**（tf32 拖累精度）。持久容器 `hgpt-rl`，GPU 1。

**Step 0 – B 纯函数测试（本地，1 分钟）**
```bash
.venv/Scripts/python.exe -m pytest tests/test_rl_train.py -q
.venv/Scripts/python.exe -m py_compile rl/reward.py rl/residual.py rl/tennis_env.py rl/train_ppo.py
```

**Step 0 – C1 几何校正（服务器，现有 ckpt 200 球 eval）**
```bash
sudo docker exec -d hgpt-rl bash -c "cd /workspace/Humanoid-GPT && JAX_DEFAULT_MATMUL_PRECISION=highest python rl/train_ppo.py --eval --ckpt storage/ckpts/rl/residual_clear_aim.pkl --num-envs 512 --n-balls 200 --seed 0 > logs/geom_calib.log 2>&1"
```
（`--eval` 路径必须先补 `train_ppo.py:192-194` 的 `landing_y/target_x/target_y` 三键，否则 KeyError。）

**Step 1 – A 前置探针（服务器，≤0.5 h，无训练）**
```bash
sudo docker exec -d hgpt-rl bash -c "cd /workspace/Humanoid-GPT && JAX_DEFAULT_MATMUL_PRECISION=highest python rl/train_ppo.py --probe --num-envs 512 --n-balls 200 --seed 0 --action-joints 22,25,26,27,28 > logs/probe_arm.log 2>&1"
```

**Step 2 – A run1 全量重训 + B reward 修复（服务器，≈15.4 h，新 ckpt 名 `residual_arm.pkl`）**
```bash
sudo docker exec -d hgpt-rl bash -c "cd /workspace/Humanoid-GPT && JAX_DEFAULT_MATMUL_PRECISION=highest python rl/train_ppo.py --train --num-envs 512 --total-iters 2000 --eval-interval 10 --eval-balls 200 --seed 0 --action-joints 22,25,26,27,28 --delta-scale 0.5 --ckpt storage/ckpts/rl/residual_arm.pkl > logs/residual_arm.log 2>&1"
```

**Step 0/2 – B 的 300 iter A/B（server，成本闸，新 ckpt 名 `residual_clear_fix.pkl`）**
```bash
sudo docker exec -d hgpt-rl bash -c "cd /workspace/Humanoid-GPT && JAX_DEFAULT_MATMUL_PRECISION=highest python rl/train_ppo.py --train --num-envs 512 --total-iters 300 --eval-interval 10 --eval-balls 200 --seed 0 --ckpt storage/ckpts/rl/residual_clear_fix.pkl > logs/residual_clear_fix.log 2>&1"
```

**Step 3 – C2 课程（可选，先 300 iter，新 ckpt 名 `residual_curric.pkl`）**
```bash
sudo docker exec -d hgpt-rl bash -c "cd /workspace/Humanoid-GPT && JAX_DEFAULT_MATMUL_PRECISION=highest python rl/train_ppo.py --train --num-envs 512 --total-iters 300 --eval-interval 10 --eval-balls 200 --seed 0 --curriculum near2full --ckpt storage/ckpts/rl/residual_curric.pkl > logs/residual_curric.log 2>&1"
```

**禁止覆盖的既有 ckpt**：`residual_ppo.pkl`（stage 1）、`residual_landing.pkl`（stage 2 old）、`residual_clear_aim.pkl`（stage 2 new）。本方案新增 ckpt 一律另开新路径。

**训练后正式 1000 球 eval（所有候选统一口径）**：
```bash
sudo docker exec -d hgpt-rl bash -c "cd /workspace/Humanoid-GPT && JAX_DEFAULT_MATMUL_PRECISION=highest python rl/train_ppo.py --eval --ckpt storage/ckpts/rl/residual_arm.pkl --num-envs 512 --n-balls 1000 --seed 0 --action-joints 22,25,26,27,28 > logs/eval_arm_1000.log 2>&1"
```
（`--eval` 必须从 ckpt 读 `action_joints/delta_scale` 重建 env，不能用 CLI 默认值，否则 δ 通道错位或形状不匹配。）

---

## 6. 开放问题与风险

1. **闭环抵消（最可能的失败模式，跨三方向）**：冻结跟踪器是闭环，PD 会把肩/肘扰动拉回。三个方向都不改跟踪器本身；A 的探针能测净落点变化（覆盖此模式），但 B/C 完全没碰它。若探针显示权限不足，正确的下一步不是叠加近端 DOF，而是**改动更小的杠杆：让跟踪器参考挥拍在 SWING 段对持拍臂（肩/肘）降低跟踪刚度**，把"权限"从机制上打开。
2. **过网判据 vs 目标域混淆**：过网 := `landing_x > 3.0`，与目标域 x∈[4,10] 无关。需要证的是"新增关节能否把球推过 x=3.0"，不是"能否落到 7–8"。方向 A 的乐观叙述存在此机制跳跃，需在探针中以 `Δlanding_x`（落点深度）直接度量。
3. **reward 饱和是横切问题**：三份草稿都指出 `r_clear` 饱和 + 未过网常数回退会压制新动作权限的学习信号。**最终选定的 full retrain 应直接带上 B 的修复，而不是先单独烧一次 2000-iter。**
4. **A 的收益区间无实测支撑**：探针前中心 +4 pp、乐观 58–64% 均为外推，且乐观上修仍依赖"探针权限可被 RL 学会"这一未证环节。
5. **探索噪声偏斜**：`log_std` 状态无关；把 6 倍物理增益差的关节纳入同一噪声尺度后，物理空间探索严重偏斜。run1 用统一 0.5 是干净消融，但可能低估 A（或高估其噪声代价）；run2 逐关节等物理权限才公平，成本翻倍。
6. **服务器 ckpt 依赖**：`residual_landing.pkl / residual_clear_aim.pkl` 在服务器 `hgpt-rl`，本地不存在——所有零成本探针/几何校正须在服务器执行，无法完全离线预演。
7. **C1 数字极易被断章引用**：headline 从 3.45→~0.9–1.3 m 极易被读成"改进 2.5 m"。三值并列（`D_full / D_geo / D_obs`）纪律必须严格执行，否则"重定义"就伪装成"改进"。
8. **未验证项**：肩/肘对落点的净权限完全未测；PD/力矩饱和（`TORQUE_LIMIT[22]=25 Nm`）未验证；C2 是否为 curriculum trap 未知；`SALT` 常量与 `DEFAULT_DELTA_SCALE_VEC` 具体值待定（实现时须补齐）。
9. **评测可比性纪律**：动作空间变化后 hit/over-net 仍同 env 可比，但须固定同 seed 与 200/1000 球口径；C2 评估必须 `curriculum=1.0` 固定全框。

---

## 7. 与 LATENT 论文的同口径对比（SR@2.5m）

在论文同口径（回球落点距目标 ≤2.5 m 的比例，SR@2.5）下，基线 ckpt `residual_clear_aim.pkl`（1000 球）
为 **15.6%（156/1000）**，过网球口径 **26.3%（136/518）**；LATENT 论文仿真正手为 **96.5%**，
差距约 **6×**（过网口径仍约 1/4）。该差距是**结构性**的（动作权限），reward 调参 / 几何重定义均无法闭合。
完整口径对照、SR 曲线、三条根因证据与复现方式见 `docs/LATENT_SR_COMPARISON.md`。

> 口径警告：本方案中的"命中率 96%"是拍-球**接触**率，**不是**落点精度，不可与论文 **96.5%** 的 SR 混用。
