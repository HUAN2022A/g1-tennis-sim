# 网球击球闭环开发交接文档

更新日期：2026-09-04
前置旧交接：`docs/TENNIS_REPLICATION_HANDOFF.md`（2026-09-03，已过时，仅作历史参考——其记录的环境/数据/场景内容大多已并入 git 与本文档）

## 1. 当前目标

在 Windows 本地以 CPU 运行 Humanoid-GPT，让 Unitree G1 在 MuJoCo 中实现**网球击球闭环**：感知来球 → 决策 → 触发挥拍 → 命中。最终路线是 LATENT 式（论文 arXiv:2603.12686）的分层系统——本仓库的 Humanoid-GPT 是低层运动跟踪器（"小脑"），在其上构建高层状态机（"大脑雏形"）。

```text
高层（本阶段：规则状态机，未来：RL/视觉语言大脑）
    -> 选择动作片段 + 控制触发时机
Humanoid-GPT（现有，低层）：状态反馈 + 参考轨迹 -> 29 电机目标 -> PD -> MuJoCo
```

## 2. 环境基线（已验证，勿动）

- 工作目录 `C:\code\Humanoid-GPT`，Windows，Python 3.12，虚拟环境 `.venv`（`uv sync --extra cpu` 创建）
- MuJoCo 3.3.7、ONNX Runtime CPU、JAX/Torch CPU
- 跟踪模型 `storage/ckpts/pns_wo_priv264.onnx`，机器人 Unitree G1 29 DoF
- 网球轨迹：`C:\Users\huan2\Documents\具身智能\LATENT\storage\data\mocap\Tennis\p1`（四条：Random_001~004，`qpos (T,36)`，Random_001 最稳）
- **数据真相**：上述轨迹来自 LATENT 论文项目（清华/Galbot），机器人同为 G1。论文笔记见 `docs/LATENT_PAPER_NOTES.md`，训练代码开源 `github.com/GalaxyGeneralRobotics/LATENT`

## 3. 已完成的工作（均提交）

| commit | 内容 | 验证结果 |
|---|---|---|
| `8418e5f` | #7 `play_track.py` 接 `--tennis`/`--racket_mass`，交互窗口看带拍跟踪 | 无界面冒烟全过 + 回归 Random_001 完成率 1.0；顺带修 `LiveRefConverter` nq=43 截断 bug |
| `fb65999` | #6 固定来球命中测试 `scripts/tennis_hit_test.py`（三阶段两回放法）+ 公共模块 `utils/tennis_common.py` | 50g 拍 14-15 m/s 挥速无穿模，反弹物理合理；**交接文档担心的穿模风险排除** |
| `b27920b` | #8 V0 站桩命中状态机 `scripts/tennis_v0_stationary.py` + 弹道预测 `utils/ball_predict.py` + 两份文档 | **20/20 = 100% 命中**（验收线 80%） |

**V0 标定值**（Random_001，50g 拍）：挥拍 onset 帧 150（site 速度 2.0 m/s 阈值自动判据）、`T_lead=0.26s`、`P_swing=(2.05,-1.97,0.55)`。

**两个必须记住的标定陷阱**（已写入 commit message）：
1. 挥拍 onset 帧根部速度 ~1.6 m/s，站桩起步必须清零，否则机器人前冲 0.75m
2. 站桩保持帧数必须与标定一致，机器人有 ~0.01m/帧漂移，帧数不一致使拍面轨迹漂移 0.35m

## 4. 进行中：V1 鲁棒命中（未提交）

**目标**：随机可行域 + 球物理域随机化 + 观测噪声下 50 球命中率 ≥ 50%，输出失误分类。

**文件**：`utils/ball_obs.py`（观测噪声 + 4 帧滑窗速度 + dropout + 延迟）、`scripts/tennis_v1_robust.py`（主脚本，26KB）。

**当前实测**（A+C 参数：feas_radius=0.06、阻力 k U(0.005,0.02)、预测侧 k=0.015）：
- σ=0mm（无噪声对照）：25/50 = 50%，恰好踩线
- σ=5mm：17-21/50 = 34-42%，未达验收线，子 agent 仍在迭代

**核心结论雏形**：失误主因是"预测偏差"而非"球不可达"——逐球数据里真值拦截偏差多在 0.02-0.06m（在 0.08m 命中圆内），但预测偏差 0.07-0.14m。观测噪声是当前最大误差源。触发时序控制得很好（误差 ±2ms）。

**误差预算框架**（V1 最有价值的产出）：命中圆 0.08m vs 三层误差（发射采样 / 空气阻力失配 / 观测噪声）——三层各贡献 ~0.04-0.10m 偏移，叠加超预算导致命中率上限被压低。

## 5. 下一步计划

### 主线（状态机闭环）

| 里程碑 | 内容 | 验收 | 依赖 |
|---|---|---|---|
| **V1**（进行中） | 随机化 + 观测噪声鲁棒命中 | ≥50% + 失误分类 | 无 |
| **V2** | 正/反手片段选择（按预测拦截点左右侧选片段） | 混合命中 ≥40% | **需先人工标注四条轨迹语义** |
| **V3** | walk policy 导航到拦截点 → 切换挥拍片段 | 移动后命中不比站桩劣化 >10pp | walk policy（`play_track` 模式 0 已有） |

### 支线

- **V2 前置人工任务**：用 `--tennis` 交互窗口看四条轨迹叠影，标注正手/反手/其它语义——机器判断不了动作语义，必须人来做
- **#2**（低优先级）：定位 Random_004 失稳帧，数据已在 `logs/tennis_export/`

### 远期：第二步 RL 微调

LATENT 式训练（腕部修正接口 + 域随机化 + PPO，8 GPU）。**本阶段的 V1 失误分类和误差预算数据正是它的立项依据**。天花板认知：LATENT 消融证明无腕部修正时正手 SR 从 96.5→82.4——无学习状态机的极限在此，属预期。

## 6. 关键源码位置

- `utils/tennis_scene.py`：网球场景加载（MjSpec 挂拍 50g）、球状态读写
- `utils/tennis_common.py`：build_tennis_sim / site_state / contact_event / set_ball 公共模块（#6/V0/V1 共用）
- `utils/ball_predict.py`：弹道预测（重力 + 可选空气阻力，下降段拦截）
- `utils/ball_obs.py`：观测噪声 + 4 帧滑窗速度（V1）
- `scripts/tennis_hit_test.py`：固定来球命中测试（#6）
- `scripts/tennis_v0_stationary.py`：V0 站桩命中状态机（#8，含标定）
- `scripts/tennis_v1_robust.py`：V1 鲁棒命中（#8，进行中）
- `deploy/play_track.py`：交互式播放器（已接 --tennis，用于人工观察/标注）
- `docs/TENNIS_STATE_MACHINE_PLAN.md`：V0-V3 方案（权威）
- `docs/LATENT_PAPER_NOTES.md`：LATENT 论文调研
- `docs/TECHNICAL_REPORT.md`：主技术文档（136 维观测、29 维动作、控制链路）

## 7. 新对话建议开场

```text
请阅读 docs/HANDOFF.md（或本文件）。当前环境：Windows + CPU + MuJoCo，G1 跟踪网球动作，50g 拍场景已验证（--racket_mass 0.05）。已完成 #6 命中测试、#7 交互播放器接 tennis、#8 V0 站桩命中（20/20）、V1 鲁棒命中（进行中，σ=5mm 时命中率 34-42% 未达 50% 验收线）。下一步优先：完成 V1 达标 → 人工标注四条轨迹正/反手语义 → V2 片段选择。保留现有改动，不要重置工作区。
```

## 8. 工作方式备注（本次会话验证有效）

- 主控台（我）负责决策 + 复核 diff/产出，子 agent（Haiku + max 思考强度）负责执行，通过 SendMessage 双向通信
- 子 agent 会主动上报方案级冲突（本次两次：T_lead 定义、V1 误差预算），决策后正确执行
- 子 agent 保留跨任务记忆（同一 agentId 会话复用），后续任务可增量下达
- API 欠费会导致子 agent 中断（403 预扣费失败），充值后 SendMessage 唤醒可从断点继续
