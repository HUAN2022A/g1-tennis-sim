"""jittable 的 MJX 批量网球环境（RL 阶段 1 · T3）：整回合 lax.scan 状态机。

把 scripts/tennis_v1_robust.py（V1 状态机，逻辑真源）逐条移植到 MJX + jax：
- step：20×1ms 子步 lax.scan 的 PD 控制（作用域 qpos[7:36]/qvel[6:35]，球
  freejoint 排除，对齐 utils/sim_mj.py L134-142）+ 力矩 clip ±TORQUE_LIMIT +
  每子步对球 3 个平动 dof 施加二次阻力（qfrc_applied replace 模式，对齐
  utils/tennis_ball.py::step_with_drag）+ 球四元数每控制步重归一化。
- reset：逐环境 sample_serve_batch（V1 同参数：feas_radius=0.06、z±0.02、
  T∈U(0.65,0.85) 帧网格量化、design_k=pred_drag_k）+ 球物理 DR
  （质量×U(0.95,1.05)（含惯量同 scale）、摩擦 U(0.4,0.7)、反弹 U(0.71,0.79)
  → solref 阻尼、阻力 k U(0.005,0.02)）；机器人复位到 ref_traj[onset]
  （reset_to_frame(onset) 语义，root 速度清零），球 park 到 (100,100,5)。
- 整回合 scan：定长 N 控制步（由 V1 launch/flight/swing 上界推出，~120），
  状态机 mode ∈ {IDLE,TRACK,ARM,SWING,DONE}（DONE=吸收态，等价 V1 的
  break）+ 掩码分支；感知 → read_fit（固定 k=pred_drag_k）→
  predict_intercept → 触发（含 trigger_bias_frames=-2）/ NO_SWING 门控
  （miss_tol 0.15 + gate 0.45s×4 帧）/ EARLY_GROUND（真值 z<0.05）；
  执行：按 mode 取 ref 帧（hold 段=onset，SWING 段 onset→swing_end 逐帧）
  → get_nn_state → 冻结策略前向（残差 δ 加在 nn_action[:,26:29]）→
  nn2motor_action → step。
- 仪表：trace 记录每控制步 ball/site 位姿 + racket-ball 接触 + mode/swing_f，
  host 侧汇总命中、峰值窗 |f_rel−peak_local|≤10 内 site-球最小距、
  τ = 球过面时刻 − 拍面峰值时刻（= (trig_step+2+peak_local)·dt，与
  logs/trigfix_* 仪表同定义）。

标定：calibrate_v1() 用 CPU 管线（utils/tennis_clips.calibrate，复用 V1
调用方式）跑一次，把 t_lead/p_swing/onset/peak_local/swing 帧范围缓存到
storage/ckpts/rl/v1_calib.npz；环境加载缓存，--calibrate 可重算。

精度：全程 f32（不得开启 x64——与 MJX 冲突；rl/ball_obs_jax 默认即 f32）。
f32 数值适配：read_fit 走本模块的 _F32BallObsBuffer / _fit_fixed_k——T2 照抄
numpy 的 GN 有限差分 eps=1e-6 在 f32 下仅 ~2 ulps，FD 雅可比噪声会破坏拟合
（速度误差实测达 0.3 m/s），故 f32 用 eps=1e-3（GN 收敛点与 eps 无关，
语义不变；f64 路径仍 1e-6，对 T2 零改动）。
性能注意（T1 发现）：本机 jax CPU batch≥16 的 f32 GEMM 偶发死锁，策略
前向默认按 4 行分块（lax.map，chunk≤4 串行内核路径）。

用法：
    uv run python rl/tennis_env.py --calibrate          # 生成标定缓存
    uv run python rl/tennis_env.py --evaluate 200       # 零残差验收评估
    uv run python rl/tennis_env.py --smoke              # 单环境冒烟
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, NamedTuple

if __package__ in (None, ""):  # 直接以脚本运行：把仓库根加入 sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np
from jax import lax
from jax import random as jrandom

from rl import ball_obs_jax as bj
from rl import frozen_policy as fp
from tracking import constants as consts
from utils.tennis_scene import load_tennis_model

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIB = REPO_ROOT / "storage/ckpts/rl/v1_calib.npz"
DEFAULT_CLIP = REPO_ROOT / "storage/data/mocap/Tennis/p1/Random_001_Tennis 001.npz"

# ---------------------------------------------------------------------------
# V1 等价常量（scripts/tennis_v1_robust.py::V1Args 默认值，逐条对齐勿改）
# ---------------------------------------------------------------------------
FREQ = 50                    # 控制频率
CTRL_DT = 1.0 / FREQ         # 0.02
SIM_DT = 0.001               # 物理子步
N_SUB = int(CTRL_DT / SIM_DT)  # 20
RACKET_MASS = 0.05           # V1 球拍质量（kg，注意不是 tennis_scene 默认 0.3）

# 层 1：发射随机化
FEAS_RADIUS = 0.06
FLIGHT_LO, FLIGHT_HI = 0.65, 0.85
SERVE_HEIGHT, SERVE_BACK = 1.0, 2.8
HOLD_FRAMES = 30

# 层 2：球物理 DR
DR_K_LO, DR_K_HI = 0.005, 0.02
DR_MASS_LO, DR_MASS_HI = 0.95, 1.05
DR_FRIC_LO, DR_FRIC_HI = 0.4, 0.7
DR_REST_LO, DR_REST_HI = 0.71, 0.79
BASE_SOLREF_DAMP = 0.2       # scene_tennis.xml 球 solref="0.01 0.2"

# 层 3：观测噪声（V1 默认）
OBS_SIGMA = 0.005
OBS_DROPOUT_P = 0.05
OBS_DELAY_FRAMES = 1
OBS_WINDOW = 4               # read_fit min_points（= V1 obs_window）
OBS_HIST_LEN = 7             # numpy BallObserver deque maxlen = delay+window+2
MAX_POINTS = 16              # read_fit max_points

# 状态机参数（V1 同款）
MISS_TOL = 0.15
ARM_HORIZON = 2.0
Z_PLANE_OFFSET = 0.0
GATE_HORIZON = 0.45
GATE_FRAMES = 4
GROUND_Z = 0.05              # EARLY_GROUND 真值阈值
PRED_DRAG_K = 0.0125         # 预测/拟合/发射设计共用阻力假设
TRIGGER_BIAS_FRAMES = -2     # 触发时序偏置（dbf965f 正式修正）

# 阶段 2（命中→落点）：球网 / 目标落点几何（对齐 scene_tennis.xml）
NET_X = 3.0                  # 球网 x 坐标（<body name="net" pos="3 0 0">）
NET_HEIGHT = 0.914           # 网高（net_band pos z=0.914）
LANDING_X_LO, LANDING_X_HI = 4.0, 10.0     # 目标落点 x 采样域（对方半场，网后 1m 到底线前）
LANDING_Y_LO, LANDING_Y_HI = -4.0, 4.0     # 目标落点 y 采样域（单打场内）

# 模式 / 结果枚举
MODE_IDLE, MODE_TRACK, MODE_ARM, MODE_SWING, MODE_DONE = 0, 1, 2, 3, 4
OUT_NONE, OUT_HIT, OUT_MISS, OUT_NOSWING, OUT_EARLY = 0, 1, 2, 3, 4

# 策略前向分块（T1 发现的 CPU GEMM batch≥16 死锁规避；≤4 走串行路径）
POLICY_CHUNK_ROWS = 4

PARK_POS = (100.0, 100.0, 5.0)   # 发射前球待命位（reset_to_frame 语义）


# ---------------------------------------------------------------------------
# 标定：CPU 管线跑一次 utils/tennis_clips.calibrate，缓存到 npz
# ---------------------------------------------------------------------------

def calibrate_v1(clip_path: str | Path = DEFAULT_CLIP,
                 calib_path: str | Path = DEFAULT_CALIB,
                 racket_mass: float = RACKET_MASS,
                 freq: int = FREQ,
                 hold_frames: int = HOLD_FRAMES,
                 force: bool = False) -> dict:
    """V1 标定并缓存。缓存命中且 force=False 时直接读缓存。

    复用 V1 的调用方式：load_segment → build_tennis_sim(racket_mass=0.05) →
    G1TrackInferFn(onnx) → utils/tennis_clips.calibrate（CPU MuJoCo 管线）。
    """
    calib_path = Path(calib_path)
    if calib_path.exists() and not force:
        return load_calib(calib_path)

    # CPU 管线依赖 onnxruntime，延迟导入（环境本体不需要）
    from tracking.infer_utils import G1TrackInferFn, g1_infer_env_config
    from tracking.policy import Args as PolicyArgs, get_policy_onnx
    from utils.tennis_clips import calibrate, load_segment
    from utils.tennis_common import build_tennis_sim

    ctrl_dt = 1.0 / freq
    print(f"[CAL] Loading segment: {clip_path}")
    ref_traj = load_segment(str(clip_path), str(consts.TRACK_XML), freq)
    traj_len = len(ref_traj["qpos"])
    print(f"[CAL] Segment length: {traj_len} frames")
    sim, site_id, ball_adr, ball_dof = build_tennis_sim(
        racket_mass=racket_mass, ctrl_dt=ctrl_dt, sim_dt=SIM_DT)
    policy = get_policy_onnx(PolicyArgs(onnx_track=str(fp.DEFAULT_ONNX)))
    infer_fn = G1TrackInferFn(g1_infer_env_config(ctrl_dt=ctrl_dt),
                              sim.mj_model, policy, privileged=False)

    print("[CAL] Calibrating (reuse V1 procedure)...")
    cal = calibrate(sim, infer_fn, ref_traj, site_id, ball_adr, ball_dof,
                    ctrl_dt, 0, hold_frames)
    peak_local = int(round(cal["t_lead"] / ctrl_dt))
    swing_end = min(cal["onset"] + peak_local + 30, traj_len - 1)
    out = {
        "t_lead": float(cal["t_lead"]),
        "p_swing": np.asarray(cal["p_swing"], dtype=np.float64),
        "onset": int(cal["onset"]),
        "peak_local": peak_local,
        "swing_end": int(swing_end),
        "traj_len": int(traj_len),
        "hold_frames": int(hold_frames),
        "freq": int(freq),
        "racket_mass": float(racket_mass),
        "site_traj_swing": np.asarray(cal["site_traj_swing"], dtype=np.float64),
    }
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(calib_path, **out)
    print(f"[CAL] onset={out['onset']}, T_lead={out['t_lead']:.3f}s, "
          f"P_swing={np.round(out['p_swing'], 3).tolist()}, "
          f"peak_local={peak_local}, swing=[{out['onset']},{out['swing_end']}]")
    print(f"[CAL] saved to {calib_path}")
    return out


def load_calib(calib_path: str | Path = DEFAULT_CALIB) -> dict:
    with np.load(Path(calib_path)) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# 环境状态 / 单步 trace
# ---------------------------------------------------------------------------

class ResidualCtx(NamedTuple):
    """T4 残差观测上下文：_step 在策略前向前组装的只读感知快照。

    由 make_obs_r（rl/residual.py）消费。字段均为步前（本控制步物理步进前）
    的量；ball_* / inter_* 来自 read_fit / predict_intercept（对观测噪声后的
    球迹拟合，与决策链同源）。rng 供残差探索噪声只读派生
    （fold_in 不推进流，不影响观测噪声序列——语义零改动）。
    """
    ball_pos: jax.Array        # (B,3) read_fit 球位置估计（世界系，最新可用观测时刻）
    ball_vel: jax.Array        # (B,3) read_fit 球速度估计（世界系）
    ready: jax.Array           # (B,) bool 拟合就绪（观测窗 >= 4）
    inter_p: jax.Array         # (B,3) 预测拦截点（z_plane 首个下降穿越；未找到 = 0）
    t_arr: jax.Array           # (B,) 预测到达时刻（秒；未找到 = 0）
    found: jax.Array           # (B,) bool 拦截点是否找到
    site_pos: jax.Array        # (B,3) 步前 racket_point site（世界系）
    root_pos: jax.Array        # (B,3) 步前根位置（世界系）
    root_yaw: jax.Array        # (B,) 步前根偏航（quat2yaw）
    ep_step: jax.Array         # (B,) int32 控制步计数（全体同相位）
    launch_step: jax.Array     # (B,) int32 本回合发射步
    mode: jax.Array            # (B,) int32 本步执行的模式 MODE_*
    active: jax.Array          # (B,) bool 本步是否有效（= ~done0；挥拍耗尽/NO_SWING
                               #   后为 False，供残差门控抑制无梯度信号的尾部扰动）
    rng: jax.Array             # (B,2) 本步 state.rng（观测噪声流 key，只读）


class EnvState(NamedTuple):
    """整回合 carry（全部带前导批量维 B，f32/int32，pytree 友好）。"""
    data: mjx.Data             # (B, ...) 批量 mjx 数据
    rng: jax.Array             # (B, 2) 观测噪声流
    obs_buf: bj.BallObsState   # (B, ...) 观测环形缓冲
    ep_step: jax.Array         # (B,) int32 控制步计数
    mode: jax.Array            # (B,) int32 MODE_*
    swing_f: jax.Array         # (B,) int32 当前挥拍参考帧
    miss_votes: jax.Array      # (B,) int32 NO_SWING 连续超差帧数
    done: jax.Array            # (B,) bool 回合结束（吸收态）
    hit: jax.Array             # (B,) bool SWING 段内拍-球接触
    outcome: jax.Array         # (B,) int32 OUT_*
    trig_step: jax.Array       # (B,) int32 触发决策步（未触发 = -1）
    last_action: jax.Array     # (B, 29) 上一帧 nn 输出（含残差）
    # ---- 逐回合参数（reset 时采样，回合内不变）----
    serve_pos: jax.Array       # (B, 3)
    serve_vel: jax.Array       # (B, 3)
    serve_target: jax.Array    # (B, 3)
    T_flight: jax.Array        # (B,)
    launch_step: jax.Array     # (B,) int32
    k_drag: jax.Array          # (B,) 真实阻力系数
    ball_mass: jax.Array       # (B,) DR 后球质量
    ball_inertia_scale: jax.Array  # (B,) 球惯量缩放（= 质量缩放）
    ball_fric: jax.Array       # (B,) 球滑动摩擦
    ball_solref1: jax.Array    # (B,) 球 solref 阻尼（反弹强度）
    # ---- T4 残差参数 carry ----
    # 残差策略参数走 scan carry 而非闭包：jax.jit 把闭包数组烤成常量，参数
    # 每迭代更新会导致整图重编译；carry 是 pytree 值语义，结构稳定即零重编译。
    # 零残差 / evaluate 路径保持 None（确定性残差函数闭包自带参数，忽略之）。
    res_params: object = None


class StepTrace(NamedTuple):
    """单控制步仪表输出（scan 堆叠后 host 侧汇总）。记录的是"本步执行时"的量。

    T4 起新增 obs / site_pos_pre / root_pos / fit_* / delta / rng 字段（纯
    记录，不参与任何决策或物理，零语义改动）：残差训练需要在 host 侧重放
    obs_r 序列与探索噪声（jax hook 在 scan 内无法侧录），鲁棒性探针需要
    非腕关节跟踪误差（obs 的 joint_qpos 分量）。
    """
    ball_pos: jax.Array        # (B, 3) 步后球心
    site_pos: jax.Array        # (B, 3) 步后 racket_point site
    contact: jax.Array         # (B,) bool 本步 racket-ball 接触
    mode: jax.Array            # (B,) int32 本步执行的模式
    swing_f: jax.Array         # (B,) int32 本步执行的参考帧
    active: jax.Array          # (B,) bool 本步是否有效（未 done）
    # ---- T4 新增（见类注释）----
    obs: jax.Array             # (B, 136) 本步喂给策略的观测（含腕 qpos/qvel/last_action）
    site_pos_pre: jax.Array    # (B, 3) 步前 racket_point site（obs_r 拦截误差分量）
    root_pos: jax.Array        # (B, 3) 步前根位置（obs_r 球相对位置分量）
    root_yaw: jax.Array        # (B,) 步前根偏航（obs_r 旋转分量；主观测不可恢复）
    fit_pos: jax.Array         # (B, 3) read_fit 球位置估计（残差 ctx 同源：步前
    fit_vel: jax.Array         # (B, 3)  obs_buf、最新 6 点窗口；零残差路径 = 0）
    fit_ready: jax.Array       # (B,) bool
    inter_p: jax.Array         # (B, 3) predict_intercept 拦截点（同上）
    t_arr: jax.Array           # (B,) 预测到达时刻（同上）
    fit_found: jax.Array       # (B,) bool
    delta: jax.Array           # (B, 3) 本步施加的腕残差（零残差 = 0）
    rng: jax.Array             # (B, 2) 本步 state.rng（残差探索噪声重放）


def _gvec(quat_wxyz: jax.Array) -> jax.Array:
    """骨盆系重力向量 gvec = -R(quat)^T e_z（对齐 infer_utils.update_state）。"""
    w, x, y, z = (quat_wxyz[..., i] for i in range(4))
    return -jnp.stack([2.0 * (x * z - w * y),
                       2.0 * (y * z + w * x),
                       1.0 - 2.0 * (x * x + y * y)], axis=-1)


def _policy_forward(policy, obs: jax.Array,
                    chunk_rows: int = POLICY_CHUNK_ROWS) -> jax.Array:
    """冻结策略批量前向（行独立 MLP）：B<=chunk 直接算，否则 lax.map 按
    chunk_rows 分块串行执行——规避本机 CPU batch>=16 GEMM 死锁路径，
    语义与整批前向完全一致（逐行独立）。"""
    B = obs.shape[0]
    if B <= chunk_rows:
        return policy(obs)
    pad = (-B) % chunk_rows
    if pad:
        obs = jnp.concatenate([obs, jnp.zeros((pad, obs.shape[1]),
                                              dtype=obs.dtype)], axis=0)
    n_chunks = obs.shape[0] // chunk_rows
    out = lax.map(policy, obs.reshape(n_chunks, chunk_rows, obs.shape[-1]))
    return out.reshape(n_chunks * chunk_rows, -1)[:B]


# ---------------------------------------------------------------------------
# 环境本体
# ---------------------------------------------------------------------------

class TennisEnv:
    """MJX 批量网球环境（V1 状态机的 jax 移植）。

    Args:
        num_envs: 批量环境数（jitted 形状，换值需重编译）。
        calib_path / clip_path: 标定缓存与动捕片段（参考轨迹源）。
        policy: load_policy() 的冻结策略；None 则加载默认 npz。
        residual_fn: jax 函数 (obs (B,136), ctx ResidualCtx, res_params) ->
            δ (B,3)，加在 nn_action[:,26:29]（右腕 3 关节）；None = 零残差
            （T4 残差训练用）。res_params 从 EnvState carry 读出——训练时用
            state._replace(res_params=params) 注入当前参数（避免 jit 闭包
            常量化导致的逐迭代重编译）；确定性评估函数可闭包参数并忽略
            res_params（此时为 None）。
        obs_sigma / obs_dropout_p: 观测噪声（运行时参数，不触发重编译）。
        n_steps: 回合定长控制步数；None = 由 V1 上界自动推导（~120）。
    """

    def __init__(self, num_envs: int = 8,
                 calib_path: str | Path = DEFAULT_CALIB,
                 clip_path: str | Path = DEFAULT_CLIP,
                 policy=None,
                 residual_fn: Callable | None = None,
                 obs_sigma: float = OBS_SIGMA,
                 obs_dropout_p: float = OBS_DROPOUT_P,
                 n_steps: int | None = None,
                 chunk_rows: int = POLICY_CHUNK_ROWS):
        self.num_envs = int(num_envs)
        self.obs_sigma = float(obs_sigma)
        self.obs_dropout_p = float(obs_dropout_p)
        self.residual_fn = residual_fn

        cal = calibrate_v1(calib_path=calib_path) if not Path(calib_path).exists() \
            else load_calib(calib_path)
        self.calib = cal
        self.onset = int(cal["onset"])
        self.peak_local = int(cal["peak_local"])
        self.swing_end = int(cal["swing_end"])
        self.t_lead = float(cal["t_lead"])
        self.p_swing = np.asarray(cal["p_swing"], dtype=np.float32)
        self.z_plane = float(self.p_swing[2] + Z_PLANE_OFFSET)
        self.traj_len = int(cal["traj_len"])
        self.swing_frames = self.swing_end - self.onset + 1

        # 回合定长：V1 max_ep_steps 的批量上界
        T_min = FLIGHT_LO - 0.02
        T_max = FLIGHT_HI + 0.02
        launch_max = max(1, HOLD_FRAMES - int((T_min - self.t_lead) * FREQ))
        self.n_steps = int(n_steps) if n_steps is not None else (
            launch_max + int(np.ceil(T_max * FREQ)) + self.swing_frames + 20)

        # ---- 模型与 id ----
        self.mj_model = load_tennis_model(racket_mass=float(cal["racket_mass"]))
        self.mj_model.opt.timestep = SIM_DT
        self.mx_model = mjx.put_model(self.mj_model)
        self.ball_adr = int(self.mj_model.joint("tennis_ball_free").qposadr[0])
        self.ball_dof = int(self.mj_model.joint("tennis_ball_free").dofadr[0])
        self.racket_geom = self.mj_model.geom("racket_collision").id
        self.ball_geom = self.mj_model.geom("tennis_ball_geom").id
        self.site_id = self.mj_model.site("racket_point").id
        self.ball_body = self.mj_model.body("tennis_ball").id
        self.gyro_adr = int(self.mj_model.sensor("gyro_pelvis").adr[0])

        # ---- 参考轨迹（f32，jnp 常量闭包进 jit）----
        from utils.tennis_clips import load_segment
        ref = load_segment(str(clip_path), str(consts.TRACK_XML), FREQ)
        assert len(ref["qpos"]) == self.traj_len
        self.ref_qpos = jnp.asarray(ref["qpos"], dtype=jnp.float32)          # (T,36)
        self.ref_kpt2gv = jnp.asarray(ref["kpt2gv_pose"], dtype=jnp.float32)  # (T,14,4,4)
        self.ref_cvel = jnp.asarray(ref["kpt_cvel_in_gv"], dtype=jnp.float32)  # (T,14,6)
        self.ref_hold = jax.tree.map(lambda x: x[self.onset],
                                     (self.ref_qpos, self.ref_kpt2gv, self.ref_cvel))

        # ---- PD / 观测器 ----
        self.kps = jnp.asarray(consts.KPs, dtype=jnp.float32)
        self.kds = jnp.asarray(consts.KDs, dtype=jnp.float32)
        self.torque_lim = jnp.asarray(consts.TORQUE_LIMIT, dtype=jnp.float32)
        self.base_ball_mass = float(self.mx_model.body_mass[self.ball_body])
        self.base_ball_inertia = self.mx_model.body_inertia[self.ball_body]
        # f32 数值适配版观测缓冲（read_fit 的 GN eps 按精度放大，见 _fit_fixed_k）
        self.buf = _F32BallObsBuffer(hist_len=OBS_HIST_LEN,
                                     pos_sigma=self.obs_sigma,
                                     dropout_p=self.obs_dropout_p,
                                     delay_frames=OBS_DELAY_FRAMES,
                                     ctrl_dt=CTRL_DT)

        # ---- 冻结策略 ----
        self.policy = policy if policy is not None else fp.load_policy()

        # ---- jit 编译 ----
        self._chunk_rows = int(chunk_rows)
        self._physics = self._make_physics()
        self._step_jit = jax.jit(self._step)
        self._reset_jit = jax.jit(self._reset)
        self._rollout_jit = jax.jit(self._rollout)

    # ------------------------------------------------------------------
    # 物理：单环境 20 子步 lax.scan（PD + clip + drag + mjx.step），vmap 批量
    # ------------------------------------------------------------------

    def _make_physics(self):
        mx, kps, kds, tlim = self.mx_model, self.kps, self.kds, self.torque_lim
        ball_dof, ball_body, ball_geom = (self.ball_dof, self.ball_body,
                                          self.ball_geom)
        base_inertia = self.base_ball_inertia

        def apply_dr(k_drag, mass, inertia_scale, fric, solref1):
            """逐环境球物理 DR（对齐 apply_ball_dr：质量/惯量/摩擦/solref）。"""
            return mx.replace(
                body_mass=mx.body_mass.at[ball_body].set(mass),
                body_inertia=mx.body_inertia.at[ball_body].set(
                    base_inertia * inertia_scale),
                geom_friction=mx.geom_friction.at[ball_geom, 0].set(fric),
                geom_solref=mx.geom_solref.at[ball_geom, 1].set(solref1),
            )

        def substeps_single(data, motor, dr):
            """单环境一个控制步 = 20 子步（对齐 step_with_drag / MJSim.step）。"""
            def body(d, _):
                # PD 只作用于机器人关节（球 freejoint 排除，sim_mj.py L134-142）
                pos_err = motor - d.qpos[7:36]
                vel_err = -d.qvel[6:35]
                torque = jnp.clip(kps * pos_err + kds * vel_err,
                                  -tlim, tlim)
                # 球阻力：qfrc_applied replace 模式（f = -k m v|v|，m 用 DR 后质量）
                f = bj.drag_force(d.qvel[ball_dof:ball_dof + 3], dr[0], dr[1])
                qfrc = d.qfrc_applied.at[ball_dof:ball_dof + 3].set(f)
                m_i = apply_dr(dr[0], dr[1], dr[2], dr[3], dr[4])
                return mjx.step(m_i, d.replace(ctrl=torque, qfrc_applied=qfrc)), None
            out, _ = lax.scan(body, data, xs=None, length=N_SUB)
            return out

        return jax.vmap(substeps_single, in_axes=(0, 0, 0))

    # ------------------------------------------------------------------
    # 观测更新（update_with_draws 的 σ/p 运行时参数版）
    # ------------------------------------------------------------------

    def _obs_update(self, state, true_pos, step, noise_u, drop_u,
                    sigma, drop_p):
        """带噪观测写入环形缓冲（掩码 = 本步 mode ∈ {TRACK, ARM}）。

        noise_u/drop_u: U[0,1) 采样；noise = -σ + 2σ·u（与 numpy
        BallObserver.update 同分布映射）；drop ⇔ has_valid & u<p，drop 时
        沿用 last_valid。"""
        drop = jnp.logical_and(state.has_valid, drop_u < drop_p)
        noise = -sigma + (2.0 * sigma) * noise_u
        cand = true_pos + noise
        obs = jnp.where(drop[..., None], state.last_valid, cand)
        last_valid = jnp.where(drop[..., None], state.last_valid, cand)
        has_valid = jnp.ones_like(state.has_valid)
        idx = jnp.arange(self.buf.hist_len)
        wmask = idx == state.ptr[..., None]
        hist_pos = jnp.where(wmask[..., None], obs[..., None, :], state.hist_pos)
        hist_step = jnp.where(
            wmask, jnp.broadcast_to(step[..., None].astype(jnp.int32),
                                    state.hist_step.shape), state.hist_step)
        ptr = (state.ptr + 1) % self.buf.hist_len
        return bj.BallObsState(hist_pos, hist_step, last_valid, ptr, has_valid)

    def _leaf_where(self, mask, a, b):
        """按批量掩码选择 pytree（mask (B,) 广播到各叶子形状）。"""
        def w(x, y):
            shape = (mask.shape[0],) + (1,) * (x.ndim - 1)
            return jnp.where(mask.reshape(shape), x, y)
        return jax.tree.map(w, a, b)

    # ------------------------------------------------------------------
    # 单控制步（状态机核心，V1 主循环逐条移植）
    # ------------------------------------------------------------------

    def _step(self, state: EnvState, sigma: jax.Array, drop_p: jax.Array,
              residual_fn=None):
        residual_fn = residual_fn or self.residual_fn
        B = self.num_envs
        onset = self.onset
        i = state.ep_step[0]                     # 全体同相位（同一控制步）
        active0 = ~state.done

        # (1) SWING 段耗尽 → 回合结束（V1 循环入口的 break）
        swing_exh = active0 & (state.mode == MODE_SWING) \
            & (state.swing_f > self.swing_end)
        done0 = state.done | swing_exh
        outcome0 = jnp.where(swing_exh,
                             jnp.where(state.hit, OUT_HIT, OUT_MISS),
                             state.outcome)
        active = ~done0

        # (2) 发射（IDLE & i == launch_step）：置球、切 TRACK、清观测缓冲
        is_launch = active & (state.mode == MODE_IDLE) \
            & (i == state.launch_step)
        quat_id = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 0.0, 0.0],
                                               jnp.float32), (B, 4))
        ball7_new = jnp.concatenate([state.serve_pos, quat_id], axis=-1)
        qpos = state.data.qpos
        qpos = qpos.at[:, self.ball_adr:self.ball_adr + 7].set(
            jnp.where(is_launch[:, None], ball7_new,
                      qpos[:, self.ball_adr:self.ball_adr + 7]))
        qvel = state.data.qvel
        vel6_new = jnp.concatenate([state.serve_vel,
                                    jnp.zeros((B, 3), jnp.float32)], axis=-1)
        qvel = qvel.at[:, self.ball_dof:self.ball_dof + 6].set(
            jnp.where(is_launch[:, None], vel6_new,
                      qvel[:, self.ball_dof:self.ball_dof + 6]))
        data = state.data.replace(qpos=qpos, qvel=qvel)
        mode = jnp.where(is_launch, MODE_TRACK, state.mode)
        obs_buf = self._leaf_where(
            is_launch, self.buf.make_init((B,)), state.obs_buf)

        # (3) 参考帧选择：SWING → traj[swing_f]/traj[swing_f+1]，其余 → hold(onset)
        use_sw = mode == MODE_SWING
        f_idx = jnp.clip(state.swing_f, 0, self.traj_len - 1)
        f_next = jnp.clip(state.swing_f + 1, 0, self.traj_len - 1)
        c_idx = jnp.where(use_sw, f_idx, onset)
        n_idx = jnp.where(use_sw, f_next, onset)
        ref_curr_q = jnp.take(self.ref_qpos, c_idx, axis=0)
        ref_next_q = jnp.take(self.ref_qpos, n_idx, axis=0)
        ref_next_kpt = jnp.take(self.ref_kpt2gv, n_idx, axis=0)
        ref_next_cvel = jnp.take(self.ref_cvel, n_idx, axis=0)

        # (3.5) 残差观测用的球拟合 + 拦截预测【T4 新增，仅残差路径】：
        # ResidualCtx 需要在策略前向之前拿到 read_fit 产物。注意窗口语义：
        # 此处读的是步前 obs_buf（标签 ≤ i-1，含最旧一格 i-7），而第 (7) 步
        # 决策读的是步后缓冲——本步 (6) 写入标签 i 时环形覆盖掉 i-7，故决策
        # 链的有效窗口是「最新 6 点」。此处用 max_points = hist_len-1 = 6
        # 取前 6 新的点，与决策链窗口一致（填满前两者窗口本就相同）。
        # 零残差路径不执行本段（决策链代码零改动，验收基线逐位不受影响）。
        fit_pos = jnp.zeros((B, 3), jnp.float32)
        fit_vel = jnp.zeros((B, 3), jnp.float32)
        fit_ready = jnp.zeros((B,), jnp.bool_)
        fit_inter = jnp.zeros((B, 3), jnp.float32)
        fit_tarr = jnp.zeros((B,), jnp.float32)
        fit_found = jnp.zeros((B,), jnp.bool_)
        if residual_fn is not None:
            fit_pos, fit_vel, _, fit_ready = self.buf.read_fit(
                obs_buf, i, k=PRED_DRAG_K, min_points=OBS_WINDOW,
                max_points=OBS_HIST_LEN - 1)
            fit_inter, fit_tarr, fit_found = bj.predict_intercept(
                fit_pos, fit_vel, self.z_plane, PRED_DRAG_K)

        # (4) 观测 → 冻结策略（+ 残差）→ 电机目标
        gyro = data.sensordata[:, self.gyro_adr:self.gyro_adr + 3]
        gvec = _gvec(data.qpos[:, 3:7])
        yaw_d, xy_d = fp.update_coord_cmd(data.qpos, ref_curr_q)
        obs = fp.get_nn_state(gyro, gvec, data.qpos, data.qvel,
                              ref_next_q, ref_next_kpt, ref_next_cvel,
                              state.last_action, yaw_d, xy_d)
        nn_action = _policy_forward(self.policy, obs, self._chunk_rows)
        site_pos_pre = data.site_xpos[:, self.site_id]   # 步前（trace 记录用）
        root_pos = data.qpos[:, :3]
        root_yaw = fp.quat2yaw(data.qpos[:, 3:7])
        if residual_fn is not None:
            # T4 残差 hook：δ 加在 nn_action[:,26:29]（右腕 roll/pitch/yaw）、
            # nn2motor_action 之前（残差随 last_action 反馈进下一帧 obs，复刻
            # LATENT 低层对腕部修正鲁棒的闭环结构）。res_params 走 EnvState
            # carry（见 EnvState 注释），支持参数热更新不重编译。
            ctx = ResidualCtx(
                ball_pos=fit_pos, ball_vel=fit_vel, ready=fit_ready,
                inter_p=fit_inter, t_arr=fit_tarr, found=fit_found,
                site_pos=site_pos_pre, root_pos=root_pos, root_yaw=root_yaw,
                ep_step=state.ep_step, launch_step=state.launch_step,
                mode=mode, active=active, rng=state.rng)
            delta = jnp.asarray(residual_fn(obs, ctx, state.res_params),
                                dtype=jnp.float32)
            nn_action = nn_action.at[:, 26:29].add(delta)
        else:
            delta = jnp.zeros((B, 3), jnp.float32)
        motor = fp.nn2motor_action(nn_action)

        # (5) 物理步（20×1ms 子步）+ 球四元数重归一化
        dr = (state.k_drag, state.ball_mass, state.ball_inertia_scale,
              state.ball_fric, state.ball_solref1)
        data = self._physics(data, motor, dr)
        bq = data.qpos[:, self.ball_adr + 3:self.ball_adr + 7]
        data = data.replace(qpos=data.qpos.at[:, self.ball_adr + 3:
                                              self.ball_adr + 7].set(
            bq / jnp.linalg.norm(bq, axis=-1, keepdims=True)))

        ball_pos = data.qpos[:, self.ball_adr:self.ball_adr + 3]
        ball_z = ball_pos[:, 2]

        # (6) 观测器更新（mode ∈ {TRACK, ARM}；发射步也更新——V1 语义）
        upd_mask = active & ((mode == MODE_TRACK) | (mode == MODE_ARM))
        rng2 = _split_keys(state.rng)                 # (B, 2, 2)
        next_rng = rng2[:, 0]
        kw = _split_keys(rng2[:, 1])                  # (B, 2, 2)
        noise_u = _uniform(kw[:, 0], (3,))            # (B, 3)
        drop_u = _uniform(kw[:, 1], ())               # (B,)
        obs_buf = jax.tree.map(
            lambda new, cur: jnp.where(
                upd_mask.reshape((-1,) + (1,) * (new.ndim - 1)), new, cur),
            self._obs_update(obs_buf, ball_pos, jnp.full((B,), i),
                             noise_u, drop_u, sigma, drop_p),
            obs_buf)

        # (7) 决策（TRACK/ARM & i > launch_step）：read_fit → 拦截预测 →
        #     NO_SWING 门控 / 触发（bias=-2）/ ARM / EARLY_GROUND
        # 【与 T3 逐行一致（未动）：决策读步后缓冲，本步新写入的标签 i 被
        #   cutoff=i-1 排除、同时环形覆盖掉最旧一格——有效窗口 = 最新 6 点；
        #   (3.5) 残差侧的拟合即按此窗口对齐，见该处注释】
        dec_mask = upd_mask & (i > state.launch_step)
        opos, ovel, _, ready = self.buf.read_fit(
            obs_buf, i, k=PRED_DRAG_K, min_points=OBS_WINDOW,
            max_points=MAX_POINTS)
        inter_p, t_arr, found = bj.predict_intercept(
            opos, ovel, self.z_plane, PRED_DRAG_K)
        dev = jnp.linalg.norm(inter_p - self.p_swing, axis=-1)
        ok = ready & found
        over = ok & (dev > MISS_TOL)
        votes_new = jnp.where(
            over, jnp.where(t_arr < GATE_HORIZON, state.miss_votes + 1, 0), 0)
        miss_votes = jnp.where(dec_mask & ok, votes_new, state.miss_votes)
        no_swing = dec_mask & ok & over & (miss_votes >= GATE_FRAMES)
        trigger = dec_mask & ok & (~over) & \
            (t_arr + TRIGGER_BIAS_FRAMES * CTRL_DT - self.t_lead
             <= 0.5 * CTRL_DT)
        arm = dec_mask & ok & (~over) & (~trigger) & \
            (t_arr <= self.t_lead + ARM_HORIZON)
        mode2 = jnp.where(no_swing, MODE_DONE,
                          jnp.where(trigger, MODE_SWING,
                                    jnp.where(arm, MODE_ARM, mode)))
        swing_f2 = jnp.where(trigger, onset, state.swing_f)
        trig_step = jnp.where(trigger, i, state.trig_step)
        early = dec_mask & (~no_swing) & (~trigger) \
            & ((mode2 == MODE_TRACK) | (mode2 == MODE_ARM)) & (ball_z < GROUND_Z)
        done1 = done0 | no_swing | early
        outcome1 = jnp.where(no_swing, OUT_NOSWING,
                             jnp.where(early, OUT_EARLY, outcome0))

        # (8) SWING 帧推进 + 命中检测（本步 mode==SWING 且 swing_f<=swing_end）
        swing_stepped = active & (mode == MODE_SWING) \
            & (state.swing_f <= self.swing_end)
        swing_f3 = jnp.where(swing_stepped, state.swing_f + 1, swing_f2)
        contact = self._racket_ball_contact(data)
        hit = state.hit | (swing_stepped & contact)

        new_state = EnvState(
            data=data, rng=next_rng, obs_buf=obs_buf,
            ep_step=state.ep_step + 1, mode=mode2, swing_f=swing_f3,
            miss_votes=miss_votes, done=done1, hit=hit, outcome=outcome1,
            trig_step=trig_step, last_action=nn_action,
            serve_pos=state.serve_pos, serve_vel=state.serve_vel,
            serve_target=state.serve_target, T_flight=state.T_flight,
            launch_step=state.launch_step, k_drag=state.k_drag,
            ball_mass=state.ball_mass,
            ball_inertia_scale=state.ball_inertia_scale,
            ball_fric=state.ball_fric, ball_solref1=state.ball_solref1,
            res_params=state.res_params,   # T4：残差参数原样透传（carry）
        )
        trace = StepTrace(ball_pos=ball_pos,
                          site_pos=data.site_xpos[:, self.site_id],
                          contact=contact, mode=mode,
                          swing_f=state.swing_f, active=active,
                          obs=obs, site_pos_pre=site_pos_pre,
                          root_pos=root_pos, root_yaw=root_yaw,
                          fit_pos=fit_pos, fit_vel=fit_vel,
                          fit_ready=fit_ready, inter_p=fit_inter,
                          t_arr=fit_tarr, fit_found=fit_found,
                          delta=delta, rng=state.rng)
        return new_state, trace

    def _racket_ball_contact(self, data) -> jax.Array:
        """racket_collision-tennis_ball_geom 接触谓词（向量化 geoms_colliding）。"""
        geom = data.contact.geom      # (B, ncon, 2)
        dist = data.contact.dist      # (B, ncon)
        pair = jnp.asarray([self.racket_geom, self.ball_geom])
        mask = ((geom == pair).all(-1) | (geom == pair[::-1]).all(-1))
        return jnp.any(mask & (dist < 0.0), axis=-1)

    # ------------------------------------------------------------------
    # reset：批量发射采样 + 球物理 DR + 机器人复位到 onset 帧
    # ------------------------------------------------------------------

    def _reset(self, rng: jax.Array):
        B = self.num_envs
        keys = jrandom.split(rng, B)                 # (B, 2)
        ks4 = jax.vmap(lambda k: jrandom.split(k, 4))(keys)   # (B, 4, 2)
        k_serve, k_dr, k_stream, k_landing = ks4[:, 0], ks4[:, 1], ks4[:, 2], ks4[:, 3]
        ks4 = jax.vmap(lambda k: jrandom.split(k, 4))(k_serve)  # (B, 4, 2)
        u_r = _uniform(ks4[:, 0], ())
        u_th = _uniform(ks4[:, 1], ())
        u_z = _uniform(ks4[:, 2], ())
        u_T = _uniform(ks4[:, 3], ())
        serve_pos, serve_vel, T_flight, target = bj.sample_serve_batch(
            jnp.asarray(self.p_swing), u_r, u_th, u_z, u_T,
            FEAS_RADIUS, FLIGHT_LO, FLIGHT_HI, SERVE_HEIGHT, SERVE_BACK,
            design_k=PRED_DRAG_K)
        # launch_step = max(1, hold - int((T - t_lead)*freq))（int = 截断）
        launch_step = jnp.maximum(
            1, HOLD_FRAMES - ((T_flight - self.t_lead) * FREQ).astype(
                jnp.int32))
        kd4 = jax.vmap(lambda k: jrandom.split(k, 4))(k_dr)    # (B, 4, 2)
        u_k = _uniform(kd4[:, 0], ())
        u_m = _uniform(kd4[:, 1], ())
        u_f = _uniform(kd4[:, 2], ())
        u_rest = _uniform(kd4[:, 3], ())
        k_drag = DR_K_LO + (DR_K_HI - DR_K_LO) * u_k
        mass_scale = DR_MASS_LO + (DR_MASS_HI - DR_MASS_LO) * u_m
        fric = DR_FRIC_LO + (DR_FRIC_HI - DR_FRIC_LO) * u_f
        rest = DR_REST_LO + (DR_REST_HI - DR_REST_LO) * u_rest

        # 目标落点采样（阶段 2 命中→落点）：对方半场随机目标（世界系 xy）
        kl2 = jax.vmap(lambda k: jrandom.split(k, 2))(k_landing)  # (B, 2, 2)
        u_lx = _uniform(kl2[:, 0], ())
        u_ly = _uniform(kl2[:, 1], ())
        target_x = LANDING_X_LO + (LANDING_X_HI - LANDING_X_LO) * u_lx
        target_y = LANDING_Y_LO + (LANDING_Y_HI - LANDING_Y_LO) * u_ly
        target_landing = jnp.stack([target_x, target_y], axis=-1)  # (B, 2)

        # 机器人复位到 onset 帧（root 速度清零），球 park 到待命位
        robot_q = jnp.broadcast_to(self.ref_qpos[self.onset], (B, 36))
        park = jnp.asarray(PARK_POS, jnp.float32)
        qpos0 = jnp.concatenate(
            [robot_q, jnp.broadcast_to(park, (B, 3)), quat_id_b(B)], axis=-1)
        d0 = mjx.make_data(self.mx_model)
        d0 = d0.replace(qpos=qpos0[0], qvel=jnp.zeros(d0.qvel.shape,
                                                      jnp.float32))
        d0 = mjx.forward(self.mx_model, d0)
        data = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (B,) + x.shape)
            if isinstance(x, jax.Array) else x, d0)

        state = EnvState(
            data=data, rng=k_stream, obs_buf=self.buf.make_init((B,)),
            ep_step=jnp.zeros((B,), jnp.int32), mode=jnp.full((B,), MODE_IDLE,
                                                              jnp.int32),
            swing_f=jnp.full((B,), self.onset, jnp.int32),
            miss_votes=jnp.zeros((B,), jnp.int32),
            done=jnp.zeros((B,), jnp.bool_), hit=jnp.zeros((B,), jnp.bool_),
            outcome=jnp.zeros((B,), jnp.int32),
            trig_step=jnp.full((B,), -1, jnp.int32),
            last_action=jnp.zeros((B, fp.NUM_ACTION), jnp.float32),
            serve_pos=serve_pos, serve_vel=serve_vel, serve_target=target,
            T_flight=T_flight, launch_step=launch_step, k_drag=k_drag,
            ball_mass=self.base_ball_mass * mass_scale,
            ball_inertia_scale=mass_scale, ball_fric=fric,
            ball_solref1=BASE_SOLREF_DAMP * (0.75 / rest) ** 2,
        )
        extras = {"serve_pos": serve_pos, "serve_vel": serve_vel,
                  "serve_target": target, "T_flight": T_flight,
                  "launch_step": launch_step, "k_drag": k_drag,
                  "target_landing": target_landing}
        return state, extras

    # ------------------------------------------------------------------
    # 整回合 lax.scan
    # ------------------------------------------------------------------

    def _rollout(self, state: EnvState, sigma: jax.Array, drop_p: jax.Array):
        def body(carry, _):
            return self._step(carry, sigma, drop_p)
        final, traces = lax.scan(body, state, xs=None, length=self.n_steps)
        traces = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traces)
        return final, traces

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def reset(self, seed: int | jax.Array = 0):
        """reset(seed) -> (EnvState, extras)。extras 含发射采样量（调试/统计）。"""
        key = seed if isinstance(seed, jax.Array) else jrandom.PRNGKey(int(seed))
        return self._reset_jit(key)

    def step(self, state: EnvState, sigma: float | None = None,
             dropout_p: float | None = None):
        """单控制步：step(state) -> (EnvState, StepTrace)（V1 主循环的一步）。"""
        sig = self.obs_sigma if sigma is None else float(sigma)
        dp = self.obs_dropout_p if dropout_p is None else float(dropout_p)
        return self._step_jit(state, jnp.float32(sig), jnp.float32(dp))

    def rollout(self, state: EnvState, sigma: float | None = None,
                dropout_p: float | None = None):
        """整回合 scan：rollout(state) -> (EnvState, traces (B, n_steps, ...))。"""
        sig = self.obs_sigma if sigma is None else float(sigma)
        dp = self.obs_dropout_p if dropout_p is None else float(dropout_p)
        return self._rollout_jit(state, jnp.float32(sig), jnp.float32(dp))

    # ------------------------------------------------------------------
    # 评估 harness：host 侧仪表汇总
    # ------------------------------------------------------------------

    def evaluate(self, n_balls: int, sigma: float | None = None,
                 seed: int = 0, residual_fn: Callable | None = None,
                 num_envs: int | None = None, verbose: bool = True) -> dict:
        """批量评估 n_balls 球：命中率 / τ 统计 / min_dist 分布。

        residual_fn: None = 零残差（环境验收）；否则 jax 函数
            (obs (B,136), ctx ResidualCtx, res_params) -> δ (B,3)（T4 残差
            策略；评估用确定性函数，res_params 恒为 None）。
        返回 dict（含逐球数组，确定性验收用同 seed 两次比对）。
        """
        if residual_fn is not None and residual_fn is not self.residual_fn:
            # 残差函数不同：换闭包重编译（零残差路径共享默认 jit）
            saved = self.residual_fn
            self.residual_fn = residual_fn
            self._rollout_jit = jax.jit(self._rollout)
            try:
                return self._evaluate_inner(n_balls, sigma, seed, verbose)
            finally:
                self.residual_fn = saved
                self._rollout_jit = jax.jit(self._rollout)
        return self._evaluate_inner(n_balls, sigma, seed, verbose,
                                    num_envs=num_envs)

    def _evaluate_inner(self, n_balls, sigma, seed, verbose, num_envs=None):
        B = self.num_envs if num_envs is None else int(num_envs)
        assert B == self.num_envs, "num_envs 与环境构造值不同需另建环境"
        n_batches = int(np.ceil(n_balls / B))
        rec = {k: [] for k in ["hit", "outcome", "tau", "min_dist", "T", "k",
                               "launch", "trig_step", "target_dev", "tau_valid"]}
        t0 = time.time()
        for b in range(n_batches):
            key = jrandom.fold_in(jrandom.PRNGKey(seed), b)
            state, extras = self.reset(key)
            final, tr = self.rollout(state, sigma=sigma)
            n_valid = min(B, n_balls - b * B)
            self._reduce_batch(final, tr, extras, n_valid, rec)
            if verbose and (b + 1) % max(1, n_batches // 10) == 0:
                nh = int(np.sum(rec["hit"]))
                print(f"  [batch {b+1}/{n_batches}] running hit {nh}/{len(rec['hit'])}")
        wall = time.time() - t0
        return self._summarize(rec, n_balls, wall, verbose)

    def _reduce_batch(self, final, tr, extras, n_valid, rec):
        """单批 trace → 逐球指标（numpy，host 侧）。"""
        tr_np = jax.tree.map(np.asarray, tr)
        fin_np = jax.tree.map(np.asarray, final)
        ex_np = jax.tree.map(np.asarray, extras)
        for e in range(n_valid):
            active = tr_np.active[e]                     # (N,)
            mode = tr_np.mode[e]
            sw_f = tr_np.swing_f[e]
            contact = tr_np.contact[e]
            ball = tr_np.ball_pos[e]                     # (N,3)
            site = tr_np.site_pos[e]
            swing = active & (mode == MODE_SWING) & (sw_f <= self.swing_end)
            hit = bool(np.any(contact & swing))
            # 峰值窗内 site-球最小距（V1 win_log 同窗口）
            win = swing & (np.abs((sw_f - self.onset) - self.peak_local) <= 10)
            d = np.linalg.norm(ball - site, axis=-1)
            min_dist = float(np.min(d[win])) if np.any(win) else np.nan
            # τ = 球过面时刻 − 拍面峰值时刻（trig_step+2+peak_local 帧）
            launch = int(ex_np["launch_step"][e])
            tau = np.nan
            z = ball[:, 2]
            for k in range(launch + 1, len(z)):
                if z[k - 1] >= self.z_plane > z[k]:     # 首个下降穿越
                    frac = (z[k - 1] - self.z_plane) / max(z[k - 1] - z[k], 1e-9)
                    t_cross = (k - 1 + 1 + frac) * CTRL_DT
                    break
            else:
                t_cross = None
            trig = int(fin_np.trig_step[e])
            if t_cross is not None and trig >= 0:
                tau = (t_cross - (trig + 2 + self.peak_local) * CTRL_DT) / CTRL_DT
            rec["hit"].append(hit)
            rec["outcome"].append(int(fin_np.outcome[e]))
            rec["tau"].append(tau)
            rec["tau_valid"].append(tau == tau)
            rec["min_dist"].append(min_dist)
            rec["T"].append(float(ex_np["T_flight"][e]))
            rec["k"].append(float(ex_np["k_drag"][e]))
            rec["launch"].append(launch)
            rec["trig_step"].append(trig)
            rec["target_dev"].append(
                float(np.linalg.norm(ex_np["serve_target"][e] - self.p_swing)))

    def _summarize(self, rec, n_balls, wall, verbose) -> dict:
        hit = np.asarray(rec["hit"])
        outcome = np.asarray(rec["outcome"])
        tau = np.asarray(rec["tau"], dtype=np.float64)
        tau_valid = tau == tau
        min_dist = np.asarray(rec["min_dist"], dtype=np.float64)
        n_hit = int(hit.sum())
        is_miss = outcome == OUT_MISS
        stats = {
            "n": len(hit), "hit_rate": n_hit / max(len(hit), 1),
            "n_hit": n_hit,
            "n_miss_swung": int((outcome == OUT_MISS).sum()),
            "n_no_swing": int((outcome == OUT_NOSWING).sum()),
            "n_early_ground": int((outcome == OUT_EARLY).sum()),
            "n_unresolved": int((outcome == OUT_NONE).sum()),
            "tau_mean": float(np.mean(tau[tau_valid])) if np.any(tau_valid) else None,
            "tau_std": float(np.std(tau[tau_valid])) if np.any(tau_valid) else None,
            "tau_n": int(tau_valid.sum()),
            "tau_mean_hit": float(np.mean(tau[tau_valid & hit]))
            if np.any(tau_valid & hit) else None,
            "min_dist_hit_mean": float(np.nanmean(min_dist[hit & ~np.isnan(min_dist)]))
            if np.any(hit & ~np.isnan(min_dist)) else None,
            "min_dist_miss_mean": float(np.nanmean(min_dist[is_miss & ~np.isnan(min_dist)]))
            if np.any(is_miss & ~np.isnan(min_dist)) else None,
            "wall_s": wall,
            "per_ball": {k: np.asarray(v) for k, v in rec.items()},
        }
        if verbose:
            fm = lambda x: "n/a" if x is None else f"{x:+.2f}"
            print(f"[EVAL] n={stats['n']} HIT {n_hit}/{stats['n']} = "
                  f"{stats['hit_rate']:.1%} | MISS_SWUNG {stats['n_miss_swung']}"
                  f" | NO_SWING {stats['n_no_swing']} | EARLY_GROUND "
                  f"{stats['n_early_ground']} | unresolved {stats['n_unresolved']}")
            print(f"  tau (cross - peak, frames): mean {fm(stats['tau_mean'])} "
                  f"± {stats['tau_std']:.2f} (n={stats['tau_n']}, "
                  f"HIT-only {fm(stats['tau_mean_hit'])})  [CPU V1 仪表基线 -0.05f]")
            print(f"  min site-ball dist: HIT {stats['min_dist_hit_mean']:.3f} m, "
                  f"MISS_SWUNG {stats['min_dist_miss_mean']:.3f} m "
                  f"[CPU V1 0.084/0.135]")
            print(f"  wall {wall:.1f}s ({wall / max(stats['n'],1) * 1000:.0f} ms/ball, "
                  f"num_envs={self.num_envs})")
        return stats


def quat_id_b(B):
    return jnp.broadcast_to(jnp.asarray([1.0, 0.0, 0.0, 0.0], jnp.float32),
                            (B, 4))


def _split_keys(keys: jax.Array) -> jax.Array:
    """批量 key split：(B,2) -> (B,2,2)（jax 0.8 split 只收单 key，vmap 批量）。"""
    return jax.vmap(jrandom.split)(keys)


def _uniform(key_row: jax.Array, shape) -> jax.Array:
    """批量 key uniform：(B,2) x shape -> (B,)+shape。"""
    return jax.vmap(lambda k: jrandom.uniform(k, shape))(key_row)


# ---------------------------------------------------------------------------
# read_fit 的 f32 数值适配（GN 有限差分 eps 按精度放大）
# ---------------------------------------------------------------------------

def _fit_fixed_k(t_obs, p_obs, k=0.0, t_eval=None, mask=None,
                 dt=bj.DEFAULT_DT, n_steps_max=1024, gn_eps=None):
    """bj.fit_ball_state_fixed_k 的 dtype 自适应 eps 版。

    T2 照抄 numpy 的 GN 有限差分 eps=1e-6：f64 下 FD 无噪（对齐 1e-9），但
    f32 下 1e-6 ≈ 参数量级的 2 ulps，FD 雅可比噪声 ~50%，拟合速度误差实测
    可达 0.3 m/s（logs/rl_t3_fit_diag2.log），观测链噪声远超 CPU f64 基线
    （NO_SWING 门控被系统性触发）。f32 用 eps=1e-3（扰动响应 ~1e-5 m ≫
    f32 量化 ~3e-7，J 噪声 ~0.3%）；GN 是 Newton 型迭代，收敛点与 eps 无关
    （eps 只影响 J 质量），语义不变。其余常数（3 次迭代 / 正则 1e-9 /
    收敛阈值 1e-6 / 二次 lstsq 初始化）与 T2 / numpy 逐项一致，复用 T2 的
    _lstsq / _sample_positions / _integrate_traj_masked。
    """
    p_obs = jnp.asarray(p_obs)
    if gn_eps is None:
        gn_eps = 1e-6 if p_obs.dtype == jnp.float64 else 1e-3
    n = t_obs.shape[-1]
    if mask is None:
        mask = jnp.ones_like(t_obs, dtype=jnp.bool_)
    m3 = mask[..., None]
    t0 = t_obs[..., 0]
    t_rel = t_obs - t0[..., None]
    t_rel_last = t_rel[..., -1]

    A = jnp.stack([jnp.ones_like(t_rel), t_rel, t_rel * t_rel], axis=-1) * m3
    Pm = p_obs * m3
    coef = bj._lstsq(A, Pm)
    p0, v0 = coef[..., 0, :], coef[..., 1, :]
    early = t_rel_last <= 0.0
    p0 = jnp.where(early[..., None], p_obs[..., -1, :], p0)
    v0 = jnp.where(early[..., None], jnp.zeros_like(v0), v0)

    gn_active = jnp.asarray(k) > 0.0
    mask_flat = jnp.repeat(mask, 3, axis=-1)

    def sample(pp, vv):
        return bj._sample_positions(pp, vv, k, t_rel, n_steps_max, dt)

    def gn_iter(carry, _):
        p0, v0, pred, done = carry
        resid = (pred - Pm).reshape(p_obs.shape[:-2] + (3 * n,)) * mask_flat
        base = jnp.concatenate([p0, v0], axis=-1)
        pert = base[..., None, :] + gn_eps * jnp.eye(6, dtype=base.dtype)
        pp = jnp.moveaxis(pert[..., :, :3], -2, 0)
        vv = jnp.moveaxis(pert[..., :, 3:], -2, 0)
        preds_p = jax.vmap(sample)(pp, vv)
        diff = (preds_p - pred) / gn_eps
        J = jnp.reshape(jnp.moveaxis(diff, 0, -1),
                        p_obs.shape[:-2] + (3 * n, 6)) * mask_flat[..., None]
        JTJ = jnp.einsum("...ji,...jk->...ik", J, J) \
            + 1e-9 * jnp.eye(6, dtype=J.dtype)
        delta = bj._lstsq(JTJ, -jnp.einsum("...ji,...j->...i", J, resid))
        nd = jnp.linalg.norm(delta, axis=-1)
        apply_ = jnp.logical_and(jnp.logical_not(done), gn_active)
        upd = jnp.where(apply_[..., None], delta, jnp.zeros_like(delta))
        p0n = p0 + upd[..., :3]
        v0n = v0 + upd[..., 3:]
        carry = (p0n, v0n, sample(p0n, v0n), jnp.logical_or(done, nd < 1e-6))
        return carry, None

    init_done = jnp.zeros_like(t0, dtype=jnp.bool_)
    (p0, v0, _, _), _ = lax.scan(
        gn_iter, (p0, v0, sample(p0, v0), init_done), xs=None, length=3)

    if t_eval is None:
        t_eval = t_obs[..., -1]
    dur = jnp.asarray(t_eval) - t0
    steps = jnp.rint(dur / dt).astype(jnp.int32)
    p_traj, v_traj = bj._integrate_traj_masked(p0, v0, k, steps,
                                               int(n_steps_max), dt)
    late = dur <= 0.0
    pos = jnp.where(late[..., None], p0, p_traj[..., -1, :])
    vel = jnp.where(late[..., None], v0, v_traj[..., -1, :])
    return pos, vel, jnp.zeros_like(t0) + k


class _F32BallObsBuffer(bj.BallObsBuffer):
    """read_fit 用 _fit_fixed_k（f32 自适应 eps）的观测缓冲。

    窗口截取逻辑（cutoff / 环形排序 / 尾部 gather / 掩码填充）与
    T2 BallObsBuffer.read_fit 逐行一致，仅拟合调用换成本地版。"""

    def read_fit(self, state, step, k=0.0, min_points=8, max_points=16,
                 t_eval=None):
        H = self.hist_len
        M = int(max_points)
        cutoff = jnp.asarray(step, dtype=jnp.int32) - self.delay_frames
        valid = jnp.logical_and(state.hist_step >= 0,
                                state.hist_step <= cutoff[..., None])
        n_avail = jnp.sum(valid, axis=-1)
        ready = n_avail >= max(4, int(min_points))
        order = jnp.argsort(
            jnp.where(valid, state.hist_step, jnp.iinfo(jnp.int32).max),
            axis=-1, stable=True)
        sorted_step = jnp.take_along_axis(state.hist_step, order, axis=-1)
        sorted_pos = jnp.take_along_axis(state.hist_pos, order[..., None],
                                         axis=-2)
        j = jnp.arange(M)
        gather = jnp.clip(n_avail[..., None] - M + j, 0,
                          jnp.maximum(n_avail - 1, 0)[..., None])
        sel_step = jnp.take_along_axis(sorted_step, gather, axis=-1)
        sel_pos = jnp.take_along_axis(sorted_pos, gather[..., None],
                                      axis=-2)
        mask_sel = j >= (M - n_avail[..., None])
        t_obs = sel_step.astype(sel_pos.dtype) * self.ctrl_dt
        t_eval_arr = t_obs[..., -1] if t_eval is None else t_eval
        pos, vel, k_est = _fit_fixed_k(
            t_obs, sel_pos, k=k, t_eval=t_eval_arr, mask=mask_sel,
            n_steps_max=self.n_fit_steps_max)
        pos = jnp.where(ready[..., None], pos, jnp.zeros_like(pos))
        vel = jnp.where(ready[..., None], vel, jnp.zeros_like(vel))
        return pos, vel, k_est, ready


# ---------------------------------------------------------------------------
# CLI：标定 / 冒烟 / 评估
# ---------------------------------------------------------------------------

def _smoke(num_envs: int = 1, n_steps: int = 10):
    """单/少环境冒烟：reset → 若干步物理与状态机（接触与传感器可用性）。"""
    env = TennisEnv(num_envs=num_envs, n_steps=max(n_steps, 1))
    state, extras = env.reset(0)
    print(f"[SMOKE] n_steps/回合 = {env.n_steps}, calib onset={env.onset} "
          f"t_lead={env.t_lead:.3f} p_swing={np.round(env.p_swing,3).tolist()}")
    print(f"[SMOKE] T={np.round(extras['T_flight'], 3)[:num_envs]} "
          f"launch={extras['launch_step'][:num_envs]} "
          f"k={np.round(extras['k_drag'], 4)[:num_envs]}")
    t0 = time.time()
    for i in range(n_steps):
        state, tr = env.step(state)
        if i % max(1, n_steps // 5) == 0:
            print(f"  step {i}: mode={np.asarray(tr.mode)[:num_envs]} "
                  f"ball_z={np.round(np.asarray(tr.ball_pos)[:, 2], 3)[:num_envs]} "
                  f"contact={np.asarray(tr.contact)[:num_envs]}")
    print(f"[SMOKE] {n_steps} steps OK in {time.time()-t0:.1f}s "
          f"(含 jit 编译); 状态机/物理/接触 可用")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibrate", action="store_true",
                    help="重算 V1 标定并写入 storage/ckpts/rl/v1_calib.npz")
    ap.add_argument("--smoke", action="store_true", help="单环境冒烟")
    ap.add_argument("--evaluate", type=int, default=None, metavar="N_BALLS",
                    help="零残差评估 N 球")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=OBS_SIGMA)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.calibrate:
        calibrate_v1(force=True)
    if args.smoke:
        _smoke()
    if args.evaluate is not None:
        env = TennisEnv(num_envs=args.num_envs)
        env.evaluate(args.evaluate, sigma=args.sigma, seed=args.seed)
    if not (args.calibrate or args.smoke or args.evaluate is not None):
        ap.print_help()


if __name__ == "__main__":
    main()
