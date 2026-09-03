# 网球动作复现交接文档

更新日期：2026-09-03

## 1. 当前目标

在 Windows 本地以 CPU 运行 Humanoid-GPT，使 Unitree G1 在 MuJoCo 中稳定跟踪、复现网球相关全身动作；后续逐步扩展为带球拍、网球与闭环击球的具身系统。

当前项目提供的是 **G1 低层运动跟踪控制器**，可类比为“小脑/运动执行层”，不是视觉语言动作（VLA）大脑。

```text
高层大脑（未来）：感知球、理解指令、规划并生成/选择动作
    -> G1 格式参考轨迹
Humanoid-GPT（现有）：状态反馈 + 参考轨迹 -> 29 个电机目标
    -> PD 控制 -> MuJoCo 或 G1 真机
```

## 2. 已完成与验证过的环境

- 工作目录：`C:\code\Humanoid-GPT`
- 操作系统：Windows
- Python：3.12.10
- 虚拟环境：`C:\code\Humanoid-GPT\.venv`
- 环境创建方式：

```powershell
uv venv --python 3.12
uv sync --extra cpu
```

- 已确认可用：MuJoCo 3.3.7、ONNX Runtime CPU provider、JAX/Torch CPU 依赖。
- 当前跟踪模型：`storage/ckpts/pns_wo_priv264.onnx`
- 机器人：Unitree G1，29 个受控关节。

## 3. 已验证的无界面 CPU 推理

仓库自带样例使用旧格式，需要 `--convert`：

```powershell
.\.venv\Scripts\python.exe -m scripts.inference `
  --onnx_track storage/ckpts/pns_wo_priv264.onnx `
  --track_dir storage/test `
  --headless `
  --device cpu `
  --convert
```

此前样例结果：完成率 `1.0000`，关键点位置 MAE 约 `0.0085 m`，关节位置 MAE 约 `0.039 rad`。

## 4. 已定位的网球轨迹与测试命令

轨迹目录：

```text
C:\Users\huan2\Documents\具身智能\LATENT\storage\data\mocap\Tennis\p1
```

其中包含四段轨迹：

- `Random_001_Tennis 001.npz`
- `Random_002_Tennis 001.npz`
- `Random_003_Tennis 001.npz`
- `Random_004_Tennis 001.npz`

它们包含已重定向到 G1 的数据：`qpos` 形状为 `(T, 36)`，即自由根节点 `7` 维加 G1 关节 `29` 维；还含 `qvel`（`(T, 35)`）与 `frequency`。

批量无界面评估：

```powershell
.\.venv\Scripts\python.exe -m scripts.inference `
  --onnx_track storage/ckpts/pns_wo_priv264.onnx `
  --track_dir 'C:\Users\huan2\Documents\具身智能\LATENT\storage\data\mocap\Tennis\p1' `
  --headless `
  --device cpu `
  --convert
```

此前一次评估的平均完成率约为 `0.9839`。轨迹稳定性会受初始化、动作本身与跟踪误差影响，不能将此数字视为所有运行都必然达到的结果。

## 5. 交互式 MuJoCo 仿真

启动命令（同时记录日志）：

```powershell
.\.venv\Scripts\python.exe -u -m deploy.play_track `
  --no-mocap `
  --track-dir 'C:\Users\huan2\Documents\具身智能\LATENT\storage\data\mocap\Tennis\p1' `
  2>&1 | Tee-Object -FilePath .\logs\tennis_sim.log
```

会打开两个窗口：

- `MuJoCo : g1 mjx flat terrain scene`：物理仿真和第三人称观察。
- `Humanoid-GPT Deploy Controller`：键盘控制和状态显示。

控制器窗口必须获得键盘焦点。按键：

- `2`、`3`、`4`、`5`：依次播放四段离线 Tennis 轨迹。
- `R`：重置。
- `0`：切回走路模式。
- `` ` ``：退出。

实时查看日志：

```powershell
Get-Content .\logs\tennis_sim.log -Wait
```

PowerShell 经 `Tee-Object` 写入时中文有时会显示乱码，但不影响仿真控制或日志语义。

## 6. 未提交的代码修改（累计）

技术细节以 `docs/TECHNICAL_REPORT.md` 5.1.4 / 5.1.5 节为准，此处只列改动清单和验证结论。

### 6.1 交互式播放（前一会话）

- `deploy/play_track.py`：相机 `elevation=-15.0`/`distance=3.5`；控制器状态面板；离线轨迹可重复按数字键重播并重置 MuJoCo；红色半透明参考叠影。
- `deploy/keyboard_cmd.py`：主进程到 Controller 窗口的文本状态同步；模式触发事件。

### 6.2 逐帧数据导出（2026-09-03 上午）

- `tracking/infer_utils.py`：`info["obs"]` 每帧写入 136 维观测向量。
- `scripts/inference.py`：新增 `--export_dir`，输出 `<stem>_frames.npz` + `<stem>_summary.csv`（字段清单见技术报告 5.1.5）。
- 验证：四条轨迹平均完成率 0.9839（与无导出一致）。Random_004 完成率 0.9357 且 `kpt_pos_errors` 峰值 0.42 m——四条中最不稳定，逐帧数据在 `logs/tennis_export/` 可定位失稳帧。

### 6.3 网球场景（2026-09-03 下午，详见第 9 节）

- 新增 `storage/assets/unitree_g1_5010/scene_tennis.xml`、`utils/tennis_scene.py`。
- `scripts/inference.py`：新增 `--tennis` / `--racket_mass`。
- `utils/sim_mj.py`：PD 力矩改为只作用于机器人关节切片 `qpos[7:36]`/`qvel[6:35]`。
- `tracking/metrics.py`：`calculate_joint_tracking_error` 同样固定 29 维切片。
- `utils/ref_ghost.py`：`set_qpos` 适配 nq=43（参考 qpos 只填机器人部分，球四元数保单位）——修复 `--tennis --show_ref_ghost` 崩溃。
- `storage/assets/unitree_g1_5010/g1_mjx_track.xml`：仅加一行挂拍注释（无实质改动）。

仓库还有用户原有的未提交改动，尤其包括 `tracking/`、`utils/sim_mj.py`、Docker 文件、`uv.lock` 与 `docs/`；后续不可随意重置或覆盖。

## 7. 数据与模型接口的准确理解

`.npz` 只是数组容器，并非任意 `.npz` 都可以输入。

当前入口可接受：

```text
qpos: (T, 36)
  = root_pos (3) + root_quat_wxyz (4) + G1 joints (29)
```

或 `root_pos`、`root_rot`、`dof_pos` 三个可拼接数组，并建议带 `frequency`。代码兼容少量历史 G1 的 23/25/27 关节格式，但不会自动把任意人体骨架、任意机器人关节顺序或 RGB 视频转成 G1。

推理链路：

```text
G1 qpos/qvel 参考轨迹
-> qpos2kpt：重采样、正运动学、关键点/速度参考
-> obs[136]：当前 G1 状态 + 下一帧参考
-> ONNX 跟踪策略
-> nn_action[29]
-> motor_targets[29]
-> PD 力矩
-> MuJoCo 物理状态
```

模型输出不是直接导出一段“最终动作视频”，而是在每个控制周期输出 G1 的 29 个目标关节位置。它可以纠正部分误差，但不能使物理不可执行的参考动作变得稳定。

## 8. “动作模仿”与“真正打网球”的范围

### 已接近具备：无球网球动作复现

目标：让 G1 稳定完成已有的准备、移动、挥拍、收拍等参考动作。

还需重点验证：

- 每段轨迹的完成率、跌倒、脚滑、关节限位和跟踪误差。
- 挥拍瞬间的关节速度、质心稳定性和手部姿态。
- 人体/动捕数据重定向到 G1 时的比例、脚底接触、手腕和腰部关节映射。
- 轨迹的时间缩放、低通平滑和动作幅度缩放。

### 尚未具备：带球的自主击球

必须补齐：

- ~~MuJoCo 的球场、球网、网球与接触/摩擦/反弹参数~~（已完成，见第 9 节）。
- ~~球拍模型，并与 G1 手部或腕部建立刚性约束~~（已完成，见第 9 节）。
- 拍面姿态、拍头速度与球到达时刻的同步。
- 球的位置、速度和预测落点输入。
- 高层动作选择与时序控制：移动、正反手选择、挥拍触发与落点目标。

若要进一步成为 VLA，还需要视觉输入、语言指令理解、视觉语言任务策略和与低层轨迹接口相连接的高层规划模块。

## 9. 带球拍/网球的网球场景（2026-09-03 第二次会话新增）

技术细节（模型布局、代码适配点、参数来源）以 `docs/TECHNICAL_REPORT.md` 5.1.4 节为准，此处记录决策与验证结论。均未提交。

**资产来源**
- 球拍挂接方式：`noitom-robotics/AdaPT` 的 `g1_rightracket.xml`（拍子作为 wrist_yaw_link 下无关节刚体，拍面用圆柱代理碰撞体，带 `racket_point` site）。
- 网球物理参数：`wfzark/tennis-robot-rl`（半径 0.0335 m，质量 0.0577 kg，`solref="0.01 0.2"` 接触参数）。

**新增文件**
- `storage/assets/unitree_g1_5010/scene_tennis.xml`：网球场景 XML（含网球、球网、球场白线；球拍不能静态声明，见下）。机器人 qpos 保持 36 维，网球 freejoint 追加在 qpos[36:43]（dofadr=35）。
- `utils/tennis_scene.py`：用 MuJoCo `MjSpec` API 动态把球拍挂到 `right_wrist_yaw_link`（`load_tennis_model(racket_mass=...)`），并添加拍-球接触对；`reset_ball()`/`get_ball_state()` 用于设置/读取球状态。

**修改文件**：见第 6.3 节清单（含 `utils/ref_ghost.py` 的 nq=43 适配——修复 `--tennis --show_ref_ghost` 崩溃，发生在录制验证视频时）。

**验证结果（四条轨迹，完成率对比）**

| 拍质量 | Random_001 | Random_002 | Random_003 | Random_004 | 平均 |
|---|---|---|---|---|---|
| 无拍（基线） | 1.000 | 1.000 | 1.000 | 0.936 | 0.984 |
| 0.05 kg | 1.000 | 1.000 | 1.000 | 0.810 | 0.953 |
| 0.30 kg | 1.000 | 0.628 | 0.096 | 0.806 | 0.632 |

结论：轻拍（50g）几乎不影响前三条，Random_004 本身就不稳定（无拍时 0.936）；0.3 kg 拍即明显退化，尤以 Random_003 崩溃最早。日志在 `logs/tennis_racket_run.log`（0.3kg）和 `logs/tennis_racket_005.log`（50g）。

**可视化验证**：`logs/tennis_racket_view.mp4`（Random_001 + 50g 拍 + 参考叠影，37 s）。四张抽帧确认场景要素齐全：右手持拍、红色叠影、网球、球网、球场白线。

**命令**

```powershell
.\.venv\Scripts\python.exe -m scripts.inference `
  --onnx_track storage/ckpts/pns_wo_priv264.onnx `
  --track_dir 'C:\Users\huan2\Documents\具身智能\LATENT\storage\data\mocap\Tennis\p1' `
  --headless --device cpu --convert --tennis --racket_mass 0.05
```

**已知限制 / 下一步**
- 球拍质量与训练分布不符是主要失稳源；真拍约 300g，需带拍微调训练（参考 AdaPT/LATENT 的 DR 训练）或先用 50g 拍做命中测试。
- 球初始位置 (2, 0, 1)，尚无来球逻辑；下一步做固定来球命中测试（用 `racket_point` site 轨迹找拍面经过点）。
- `deploy/play_track.py` 尚未接 `--tennis`；`convert` 用的 `TRACK_XML` 无拍不影响 obs（观测只用 FK 关键点，不含拍）。
- 网球与拍接触在 sim_dt=0.001 下高速球可能穿模，必要时减小 timestep 或加 margin。

## 10. 推荐后续实施顺序（当前状态）

1. ~~对四条 Tennis 轨迹逐条运行无界面评估并保存结果~~（已完成，见第 6.2 节；Random_004 最不稳定）。
2. 使用逐帧导出数据（`logs/tennis_export/`）定位 Random_004 的失稳帧和对应关节；必要时用交互式仿真观察红色参考叠影复核。
3. ~~增加推理轨迹导出~~（已完成，`--export_dir`，技术报告 5.1.5）。
4. 明确并校验网球动作数据的来源、关节映射与动作语义；补充正手、反手、发球、移动等可执行轨迹。
5. ~~在 MuJoCo XML 中增加球拍~~（已完成，见第 9 节）；带 50g 拍验证通过，300g 需微调训练。
6. 做固定来球命中测试：跟踪轨迹时用 `racket_point` site 轨迹找拍面经过点，把球设置到该位置验证反弹。
7. 给 `deploy/play_track.py` 接 `--tennis`，交互式观察带拍跟踪。
8. 再做高层状态机：基于球状态选择已有动作片段并控制触发时机。
9. 最后才接入视觉、语言和可生成 G1 参考轨迹的“大脑”模块。

## 11. 关键源码位置

- `README.md`：推理格式和基础命令（原仓库作者维护，勿改）。
- `docs/TECHNICAL_REPORT.md`：主技术文档——136 维观测、29 维动作、控制链路，以及网球场景（5.1.4）与逐帧导出（5.1.5）的权威说明。
- `tracking/convert_qpos2kpt.py`：`qpos` 到关键点参考的转换与格式限制。
- `tracking/infer_utils.py`：观测构造、ONNX 推理、残差动作到电机目标的转换。
- `utils/tennis_scene.py`：网球场景加载（MjSpec 挂拍）、球状态读写。
- `utils/ref_ghost.py`：参考叠影渲染（已适配 nq=43 场景）。
- `storage/assets/unitree_g1_5010/scene_tennis.xml`：带球拍/网球/球场的场景 XML。
- `deploy/play_track.py`：交互式离线轨迹播放、仿真主循环。
- `deploy/keyboard_cmd.py`：Controller 键盘窗口和状态面板。
- `logs/tennis_sim.log`：最近一次交互式仿真日志。

## 12. 新对话的建议开场指令

```text
请阅读 docs/TENNIS_REPLICATION_HANDOFF.md。以其中的 Windows + CPU + MuJoCo 环境为基线（球拍/网球场景见第 9 节，--tennis --racket_mass 0.05 已验证），下一步做固定来球命中测试：用 racket_point site 轨迹定位拍面经过点，把网球放到该处验证拍-球碰撞与反弹方向。保留现有未提交改动，不要重置工作区。
```
