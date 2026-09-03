import jax
import numpy as np
import mujoco.viewer
import mujoco.mjx as mjx
from flax import struct
import onnxruntime as rt
from ml_collections import config_dict
from scipy.spatial.transform import Rotation as R

from tracking import constants as consts
from tracking.constants import ACTION_JOINT_NAMES, OBS_JOINT_NAMES, KPT_NAMES
from utils.transforms_np import (
    batch_base2navi,
    quat2mat,
    quat2yaw,
    WarpPi
)
from utils.sim_mj import (
    MJSim,
    geoms_colliding as mj_coll,
    get_sensor_data as mj_sensor
)
from utils.sim_mjx import (
    geoms_colliding as mjx_coll,
    get_sensor_data as mjx_sensor
)

# 指数移动平均（EMA）平滑系数，用于平滑参考轨迹的 qpos
# 0.8 表示 80% 历史值 + 20% 当前值，有效减少动捕数据的噪声
EMA_ALPHA = 0.8


def g1_infer_env_config(ctrl_dt=0.02) -> config_dict.ConfigDict:
    """轻量级推理/部署环境配置（不包含训练特定字段）

    Args:
        ctrl_dt: 控制周期，默认 0.02s（50Hz），即每 20ms 发送一次控制指令

    Returns:
        包含控制参数的配置字典：
        - ctrl_dt: 控制周期
        - action_scale: 动作缩放因子，限制策略网络输出的动作幅度
        - soft_joint_pos_limit_factor: 软关节限位因子，0.9 表示使用 90% 的物理限位范围
    """
    return config_dict.create(
        ctrl_dt=ctrl_dt,
        action_scale=0.25,
        soft_joint_pos_limit_factor=0.90,
    )


def apply_ema_qpos(qpos, alpha=EMA_ALPHA):
    """对参考轨迹的 qpos 应用指数移动平均（EMA）平滑

    平滑公式：
        smoothed[0] = qpos[0]
        smoothed[t] = smoothed[t-1] * alpha + qpos[t] * (1 - alpha)

    特别处理四元数（索引 3:7）的符号连续性和归一化：
    - 四元数 q 和 -q 表示相同旋转，需要确保相邻帧符号一致
    - 平滑后重新归一化，保持单位四元数约束

    Args:
        qpos: np.ndarray，形状 (T, D)，其中 D >= 7
              前 3 维是位置，3:7 是四元数 (w,x,y,z)，7+ 是关节角度
        alpha: EMA 系数，0.8 表示 80% 历史值 + 20% 当前值

    Returns:
        平滑后的 qpos 数组，形状与输入相同
    """
    T = len(qpos)
    if T <= 1:
        return qpos.copy()

    smoothed = np.empty_like(qpos)
    smoothed[0] = qpos[0]

    for t in range(1, T):
        q_prev = smoothed[t - 1, 3:7]
        q_curr = qpos[t, 3:7]
        if np.dot(q_prev, q_curr) < 0:
            q_curr = -q_curr

        smoothed[t, :3] = smoothed[t - 1, :3] * alpha + qpos[t, :3] * (1 - alpha)
        smoothed[t, 7:] = smoothed[t - 1, 7:] * alpha + qpos[t, 7:] * (1 - alpha)

        q_blended = q_prev * alpha + q_curr * (1 - alpha)
        q_norm = np.linalg.norm(q_blended)
        smoothed[t, 3:7] = q_blended / max(q_norm, 1e-8)

    return smoothed


# 动作空间维度：29 个关节（Unitree G1 的全身自由度）
# 包括：左腿 6 + 右腿 6 + 腰部 3 + 左臂 7 + 右臂 7 = 29
NUM_ACTION = 29  # len(ACTION_JOINT_NAMES)

# 观测空间维度：136 维
# 组成：3(陀螺仪) + 3(重力向量) + 29(关节位置) + 29(关节速度) + 29(上一帧动作)
#       + 29(参考关节位置) + 1(参考根节点高度) + 3(参考重力向量) + 6(参考根节点速度)
#       + 2(偏航角指令 cos/sin) + 2(XY 位置指令) = 136
NUM_STATE = 136  # 3+3+29+29+29+29+1+3+6+2+2 (gyro+gvec+jnt_qpos+jnt_qvel+last_act+ref_jnt+height+gvec+cvel+yaw+xy)


@struct.dataclass
class State:
    mj_data: mujoco.MjData = None
    mjx_data: mjx.Data = None
    rng: jax.Array = None
    info: dict = None


class G1TrackMjSim(MJSim):
    """G1 机器人运动跟踪的 MuJoCo 仿真器封装

    继承自 MJSim，添加了 G1 特定的 PD 控制参数和初始化配置。

    Attributes:
        kps: 位置环 PD 增益（比例增益），形状 (29,)
        kds: 速度环 PD 增益（微分增益），形状 (29,)
        torque_limit: 各关节的力矩限制，形状 (29,)
        init_qpos: 初始关节位置，形状 (36,) = 7(基座) + 29(关节)
    """

    def __init__(
        self,
        init_qpos,
        headless=False,
        ctrl_dt=0.02,
        sim_dt=0.001,
        xml_path=str(consts.ROOT_PATH / "scene_mjx_track.xml"),
    ):
        """初始化 G1 跟踪仿真器

        Args:
            init_qpos: 初始关节位置，用于重置仿真状态
            headless: 是否无头模式（不启动可视化窗口）
            ctrl_dt: 控制周期，默认 0.02s（50Hz）
            sim_dt: 物理仿真步长，默认 0.001s（1ms）
                   每个控制周期内仿真 20 步（ctrl_dt / sim_dt = 20）
            xml_path: MuJoCo XML 模型文件路径
        """
        super().__init__(xml_path=xml_path, ctrl_dt=ctrl_dt, sim_dt=sim_dt, headless=headless)
        self.kps = np.float32(consts.KPs)
        self.kds = np.float32(consts.KDs)
        self.torque_limit = np.float32(consts.TORQUE_LIMIT)
        self.init_qpos = np.float32(init_qpos)


class G1TrackInferFn():
    """G1 机器人运动跟踪的推理函数封装

    核心功能：
    1. 从 MuJoCo 仿真状态或真实机器人传感器构建观测向量
    2. 调用 ONNX 策略网络进行推理
    3. 将网络输出转换为电机目标位置

    支持三种推理模式：
    - infer_onnx: 标准仿真推理（需要完整的 MuJoCo 状态）
    - infer_onnx_real: 真机推理（只需要 IMU 和关节编码器数据）
    - infer_onnx_real_fast: 真机快速推理（零内存分配，使用 IOBinding）

    Attributes:
        env_config: 环境配置（控制周期、动作缩放等）
        mj_model: MuJoCo 模型
        nn_policy: ONNX 推理会话（MLP 或 Transformer）
        num_envs: 并行环境数量（用于 MJX GPU 加速）
        use_mjx: 是否使用 MuJoCo MJX（JAX GPU 加速）
        privileged: 是否使用特权信息（如全局位置、关键点速度等）
    """

    def __init__(
        self,
        env_config,
        mj_model: mujoco.MjModel,
        nn_policy: rt.InferenceSession,
        num_envs: int = 1,
        use_mjx: bool = False,
        privileged: bool = False,
    ):
        """初始化推理函数

        Args:
            env_config: 环境配置字典（ctrl_dt, action_scale 等）
            mj_model: MuJoCo 模型，用于正运动学计算和坐标变换
            nn_policy: ONNX 推理会话，输入观测向量，输出 29 维动作
            num_envs: 并行环境数量，MJX 模式下可 > 1
            use_mjx: 是否使用 MJX（JAX 加速的 MuJoCo）
            privileged: 是否使用特权信息（真机部署时通常为 False）
        """
        self.env_config = env_config
        self.mj_model = mj_model
        self.nn_policy = nn_policy
        self.num_envs = num_envs
        self.use_mjx = use_mjx
        self.privileged = privileged

        # 初始化控制相关的索引映射
        # ctrl_id_act: 动作关节的执行器 ID（29 个）
        self.ctrl_id_act = np.int32(
            [self.mj_model.actuator(n).id for n in ACTION_JOINT_NAMES]
        )
        # ctrl_id_obs: 观测关节的执行器 ID（通常与动作关节相同）
        self.ctrl_id_obs = np.int32(
            [self.mj_model.actuator(n).id for n in OBS_JOINT_NAMES]
        )
        # dof_ids_g1: G1 所有关节的自由度索引（用于 qvel）
        self.dof_ids_g1 = np.int32(
            np.hstack([self.mj_model.joint(n).dofadr for n in consts.MotorName.FULL])
        )
        # qpos_ids_g1: G1 所有关节的位置索引（用于 qpos）
        self.qpos_ids_g1 = np.int32(
            np.hstack([self.mj_model.joint(n).qposadr for n in consts.MotorName.FULL])
        )

        # 默认关节位置（nominal qpos），用于计算残差动作
        # 形状：(num_envs, 29)，所有环境共享相同的默认站姿
        self._nom_jnt_qpos = np.full(
            (self.num_envs, consts.NUM_JOINT), consts.DEFAULT_QPOS[7:]
        )
        # 动作缩放因子：根据各关节的力矩限制和 PD 增益计算
        self.act_scale = np.array(consts.ACTION_SCALE)
        self.dt = env_config.ctrl_dt

        # 计算软关节限位（soft joint limits）
        # 使用 90% 的物理限位范围，避免触及硬限位导致损坏
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self.env_config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self.env_config.soft_joint_pos_limit_factor

        # 获取关键部位的 ID，用于后续的状态提取和坐标变换
        self.site_id_chest = self.mj_model.site("chest").id
        self.site_ids_feet = np.array(
            [self.mj_model.site(name).id for name in consts.FEET_SITES]
        )
        self.geom_id_floor = self.mj_model.geom("floor").id
        self.site_id_torso_imu = self.mj_model.site("imu_in_torso").id
        self.site_id_pelvis_imu = self.mj_model.site("imu_in_pelvis").id
        self.geom_ids_left_feet = np.array(
            [self.mj_model.geom(name).id for name in consts.LEFT_FEET_GEOMS]
        )
        self.geom_ids_right_feet = np.array(
            [self.mj_model.geom(name).id for name in consts.RIGHT_FEET_GEOMS]
        )
        # 关键点（Keypoint）的 body ID，共 14 个（骨盆、膝盖、脚踝、肩膀、手肘、手腕等）
        self.body_ids_kpt_full = np.array([self.mj_model.body(n).id for n in KPT_NAMES])
        self.kpt_id_pelvis = KPT_NAMES.index("pelvis")
        self.kpt_id_torso = KPT_NAMES.index("torso_link")
        self.kpt_id_peltor = np.hstack([self.kpt_id_pelvis, self.kpt_id_torso])

        self.num_jnt = len(consts.DEFAULT_QPOS[7:])
        self.num_kpt = len(KPT_NAMES)

        # 初始化信息字典，存储当前状态和中间计算结果
        # 这些信息会被 get_nn_state() 用于构建观测向量
        self.info = {
            "step": 0,  # 当前步数
            "nn_action": np.zeros((self.num_envs, NUM_ACTION), dtype=np.float32),  # 神经网络原始输出
            "gyro_pelvis": np.zeros((self.num_envs, 3)),  # 骨盆陀螺仪数据（角速度）
            "gvec_pelvis": np.zeros((self.num_envs, 3)),  # 骨盆坐标系下的重力向量
            "linvel_pelvis": np.zeros((self.num_envs, 3)),  # 骨盆线速度（局部坐标系）
            "qpos": np.zeros((self.num_envs, self.mj_model.nq)),  # 完整关节位置
            "qvel": np.zeros((self.num_envs, self.mj_model.nv)),  # 完整关节速度
            "last_action": np.zeros((self.num_envs, len(self.ctrl_id_act))),  # 上一帧动作
            "motor_targets": self._nom_jnt_qpos.copy(),  # 电机目标位置
            # 调试导出：每帧观测向量（拼接/调用 get_nn_state 后的结果）
            "obs": np.zeros((self.num_envs, NUM_STATE), dtype=np.float32),
            # 提示状态（hint state）
            "navi_pelvis_rpy": np.zeros((self.num_envs, 3)),  # 导航坐标系下的骨盆姿态（RPY）
            "navi_torso_rpy": np.zeros((self.num_envs, 3)),  # 导航坐标系下的躯干姿态（RPY）
            "feet_contact": np.zeros((self.num_envs, 2)),  # 左右脚接触状态（-1=空中，1=接触）
            # 关键点实际状态
            "acu_kpt2gv_pose": np.full((self.num_envs, self.num_kpt, 4, 4), np.eye(4)),  # 关键点在重力坐标系下的位姿
            "acu_kpt_cvel_in_gv": np.zeros((self.num_envs, self.num_kpt, 6)),  # 关键点在重力坐标系下的速度
            # 关键点参考状态（注释掉的字段在 privileged 模式下使用）
            # "ref_next_navi_vel": np.zeros((self.num_envs, 3)),
            # "next_jnt_qpos_res": np.zeros((self.num_envs, self.num_jnt)),
            # "next_jnt_qvel_res": np.zeros((self.num_envs, self.num_jnt)),
            # "ref_next_kpt_npose": np.full(
            #     (self.num_envs, self.num_kpt, 4, 4), np.eye(4)
            # ),
            # "ref_next_kpt_cvel": np.zeros((self.num_envs, self.num_kpt, 6)),
        }

    def infer_onnx(self, state: State, ref_state: np.ndarray | dict) -> np.ndarray:
        """标准仿真推理流程

        流程：
        1. 更新状态：从 MuJoCo 提取传感器数据、关键点位置等
        2. 构建观测向量：根据 privileged 标志选择观测类型
        3. ONNX 推理：策略网络输出 29 维动作
        4. 动作转换：将网络输出转换为电机目标位置

        Args:
            state: MuJoCo 仿真状态（包含 mj_data）
            ref_state: 参考状态字典，包含：
                - ref_curr: 当前帧参考轨迹（qpos, kpt2gv_pose 等）
                - ref_next: 下一帧参考轨迹（可选，用于预测）

        Returns:
            motor_targets: 电机目标位置，形状 (num_envs, 29)
        """
        ref_next = ref_state.get("ref_next", ref_state["ref_curr"])
        self.update_state(state, ref_state)
        if self.privileged:
            # 特权模式：使用额外的全局信息（关键点速度、相对位姿等）
            obs = self.get_nn_priv_state(self.info, ref_next, self.info["last_action"])
        else:
            # 标准模式：只使用机载传感器可获取的信息
            obs = self.get_nn_state(self.info, ref_next, self.info["last_action"])

        self.info["obs"] = obs.copy()
        nn_action = self.nn_policy.infer(obs)

        motor_targets = self.nn2motor_action(nn_action)
        self.info["motor_targets"] = motor_targets.copy()

        self.info["step"] += 1
        self.info["nn_action"] = nn_action
        self.info["last_action"] = nn_action.copy()
        return motor_targets

    def infer_onnx_real(
        self,
        root_quat: np.ndarray,
        root_gyro: np.ndarray,
        jnt_qpos: np.ndarray,
        jnt_qvel: np.ndarray,
        ref_state: dict,
    ) -> np.ndarray:
        """轻量级真机推理 —— 无需机器人端的 mj_forward() 计算

        对于非特权策略，观测只需要直接的传感器读数（陀螺仪、重力向量、关节位置/速度）
        加上参考轨迹的正运动学数据。机器人端的正运动学（xpos, xmat, cvel）不是必需的，
        因此我们完全跳过昂贵的 mj_forward() 调用，大幅降低推理延迟。

        Args:
            root_quat: 基座四元数 (w, x, y, z)，形状 (4,)
            root_gyro: 基座陀螺仪数据（角速度），形状 (3,)
            jnt_qpos: 关节位置，形状 (29,)
            jnt_qvel: 关节速度，形状 (29,)
            ref_state: 参考状态字典，包含 ref_curr 和 ref_next

        Returns:
            motor_targets: 电机目标位置，形状 (1, 29)
        """
        ref_next = ref_state.get("ref_next", ref_state["ref_curr"])

        # 构建完整的 qpos / qvel 数组用于 update_coord_cmd（无需正运动学）
        qpos = np.zeros((1, self.mj_model.nq), dtype=np.float32)
        qpos[0, :3] = [0.0, 0.0, 0.78]  # 基座位置固定为默认站姿高度
        qpos[0, 3:7] = root_quat
        qpos[0, 7:] = jnt_qpos

        qvel = np.zeros((1, self.mj_model.nv), dtype=np.float32)
        qvel[0, 3:6] = root_gyro
        qvel[0, 6:] = jnt_qvel

        # 从四元数计算重力向量（纯数学计算，无需正运动学）
        pelvis2world_rot = quat2mat(root_quat[None])  # (1, 3, 3)
        gvec_pelvis = -pelvis2world_rot.transpose(0, 2, 1)[..., 2]  # (1, 3)

        # 填充 get_nn_state 所需的 info 字段
        self.info["gyro_pelvis"][:] = root_gyro[None]
        self.info["gvec_pelvis"][:] = gvec_pelvis
        self.info["qpos"][:] = qpos
        self.info["qvel"][:] = qvel

        # 对齐 yaw_d / xy_d 计算与 update_coord_cmd（与 infer_onnx 相同）
        self.update_coord_cmd(ref_state)

        # 构建观测向量并运行 ONNX 推理（使用 ref_next 的参考数据，与训练一致）
        obs = self.get_nn_state(self.info, ref_next, self.info["last_action"])
        nn_action = self.nn_policy.infer(obs)

        motor_targets = self.nn2motor_action(nn_action)
        self.info["motor_targets"] = motor_targets.copy()

        self.info["step"] += 1
        self.info["nn_action"] = nn_action
        self.info["last_action"] = nn_action.copy()
        return motor_targets

    # ------------------------------------------------------------------
    # 快速部署路径：预分配缓冲区 + IOBinding（CPU）
    # 用于真机实时控制，消除每步的内存分配开销
    # ------------------------------------------------------------------

    def _ensure_fast_buffers(self) -> None:
        """延迟分配快速推理所需的临时缓冲区和 IO 绑定

        只在第一次调用 infer_onnx_real_fast() 时执行，后续调用复用这些缓冲区。
        主要优化：
        1. 预分配 qpos/qvel/gvec 缓冲区，避免每步分配新数组
        2. 预组合观测索引，将两次索引操作合并为一次
        3. 如果策略支持 IOBinding，绑定输入/输出缓冲区，实现零拷贝推理
        """
        if getattr(self, "_fast_ready", False):
            return
        nq, nv = self.mj_model.nq, self.mj_model.nv
        ne = self.num_envs
        self._qpos_buf = np.zeros((ne, nq), dtype=np.float32)
        self._qpos_buf[:, :3] = (0.0, 0.0, 0.78)  # 基座位置固定
        self._qvel_buf = np.zeros((ne, nv), dtype=np.float32)
        self._gvec_buf = np.empty((ne, 3), dtype=np.float32)

        # 组合索引：预组合 qpos_ids_g1[ctrl_id_obs]，使得
        # info["qpos"][:, idx] 每步只需一次花式索引，而不是两次索引操作 + 临时数组
        self._qpos_obs_ids = self.qpos_ids_g1[self.ctrl_id_obs]
        self._qvel_obs_ids = self.dof_ids_g1[self.ctrl_id_obs]
        self._nom_obs = np.ascontiguousarray(
            self._nom_jnt_qpos[:, self.ctrl_id_obs], dtype=np.float32
        )

        # 如果策略支持 IOBinding（CPU/CUDA EP），绑定 ORT 输入缓冲区
        if hasattr(self.nn_policy, "bind_input_buffer"):
            self._obs_buf = self.nn_policy.bind_input_buffer(
                (ne, NUM_STATE)
            )
            self._use_iobinding = True
        else:
            self._obs_buf = np.zeros((ne, NUM_STATE), dtype=np.float32)
            self._use_iobinding = False

        # 预计算观测向量的切片布局（匹配 get_nn_state 的 hstack 顺序）
        # 3 + 3 + 29 + 29 + 29 + 29 + 1 + 3 + 6 + 2 + 2 = 136
        s = 0
        def _adv(n: int) -> slice:
            nonlocal s
            sl = slice(s, s + n); s += n; return sl
        self._sl_gyro = _adv(3)      # 陀螺仪
        self._sl_gvec = _adv(3)      # 重力向量
        self._sl_qposd = _adv(self.num_jnt)  # 关节位置误差
        self._sl_qvel = _adv(self.num_jnt)   # 关节速度
        self._sl_lastact = _adv(NUM_ACTION)  # 上一帧动作
        self._sl_refqd = _adv(self.num_jnt)  # 参考关节位置
        self._sl_height = _adv(1)    # 参考根节点高度
        self._sl_refgvec = _adv(3)   # 参考重力向量
        self._sl_refcvel = _adv(6)   # 参考根节点速度
        self._sl_yaw = _adv(2)       # 偏航角指令 (cos, sin)
        self._sl_xy = _adv(2)        # XY 位置指令
        assert s == NUM_STATE, (s, NUM_STATE)
        self._fast_ready = True

    def _fill_obs_inplace(
        self,
        ref_next: dict,
        last_action: np.ndarray,
    ) -> None:
        """将观测向量就地写入 self._obs_buf（零内存分配）

        精确镜像 get_nn_state() 的逻辑，但使用预计算的切片直接写入，
        而不是构建中间数组 + np.hstack。这是实时控制的关键优化。

        Args:
            ref_next: 下一帧参考状态（包含 qpos, kpt2gv_pose, kpt_cvel_in_gv）
            last_action: 上一帧的动作，形状 (num_envs, 29)
        """
        info = self.info
        buf = self._obs_buf
        nom_obs = self._nom_obs

        # 本体感知：陀螺仪和重力向量
        buf[:, self._sl_gyro] = info["gyro_pelvis"]
        buf[:, self._sl_gvec] = info["gvec_pelvis"]

        # 关节状态：使用预组合的观测索引进行单次花式索引，
        # 然后就地减去缓存的默认关节位置
        np.subtract(
            info["qpos"][:, self._qpos_obs_ids], nom_obs,
            out=buf[:, self._sl_qposd],
        )
        buf[:, self._sl_qvel] = info["qvel"][:, self._qvel_obs_ids]
        buf[:, self._sl_lastact] = last_action

        # 参考运动：参考关节位置（相对于默认位置）
        np.subtract(
            ref_next["qpos"][:, 7:][:, self.ctrl_id_obs], nom_obs,
            out=buf[:, self._sl_refqd],
        )
        # 参考根节点高度（Z 坐标）
        ref_root_pose = ref_next["kpt2gv_pose"][:, 0]
        buf[:, self._sl_height] = ref_root_pose[:, 2, 3:4]
        # 参考重力向量：-ref_root_pose[:, :3, :3].T[..., 2] == -ref_root_pose[:, 2, :3]
        np.negative(ref_root_pose[:, 2, :3], out=buf[:, self._sl_refgvec])
        # 参考根节点速度（6 维：线速度 + 角速度）
        buf[:, self._sl_refcvel] = ref_next["kpt_cvel_in_gv"][:, 0]

        # 全局指令：偏航角（编码为 cos/sin）和 XY 位置
        yaw_d = info["yaw_d"]
        buf[:, self._sl_yaw.start] = np.cos(yaw_d)
        buf[:, self._sl_yaw.start + 1] = np.sin(yaw_d)
        buf[:, self._sl_xy] = info["xy_d"]

    def infer_onnx_real_fast(
        self,
        root_quat: np.ndarray,
        root_gyro: np.ndarray,
        jnt_qpos: np.ndarray,
        jnt_qvel: np.ndarray,
        ref_state: dict,
    ) -> np.ndarray:
        """与 infer_onnx_real() 相同，但零每步内存分配

        预分配的临时缓冲区 + IOBinding 消除了 ONNX 调用的稳态延迟和长尾抖动。
        在 Jetson Orin 上，50Hz 控制循环的推理延迟从 ~2ms 降至 ~0.4ms。
        （详见 deploy/onboard_deploy/bench_online.py）

        Args:
            root_quat: 基座四元数 (w, x, y, z)，形状 (4,)
            root_gyro: 基座陀螺仪数据，形状 (3,)
            jnt_qpos: 关节位置，形状 (29,)
            jnt_qvel: 关节速度，形状 (29,)
            ref_state: 参考状态字典

        Returns:
            motor_targets: 电机目标位置，形状 (1, 29)
        """
        self._ensure_fast_buffers()
        ref_next = ref_state.get("ref_next", ref_state["ref_curr"])

        # 就地更新 qpos / qvel 临时缓冲区（基座 xy/z 保持默认站姿）
        self._qpos_buf[0, 3:7] = root_quat
        self._qpos_buf[0, 7:] = jnt_qpos
        self._qvel_buf[0, 3:6] = root_gyro
        self._qvel_buf[0, 6:] = jnt_qvel

        # 快速纯 NumPy 四元数 -> 重力向量转换（避免 scipy Rotation 的开销）
        # gvec = -R(quat).T @ [0,0,1] = -[2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)]
        w, x, y, z = root_quat
        self._gvec_buf[0, 0] = -2.0 * (x * z - w * y)
        self._gvec_buf[0, 1] = -2.0 * (y * z + w * x)
        self._gvec_buf[0, 2] = -(1.0 - 2.0 * (x * x + y * y))

        info = self.info
        info["gyro_pelvis"][0] = root_gyro
        info["gvec_pelvis"][:] = self._gvec_buf
        info["qpos"][:] = self._qpos_buf
        info["qvel"][:] = self._qvel_buf

        self.update_coord_cmd(ref_state)

        # 就地构建观测向量到（可能已绑定 IOBinding 的）缓冲区
        self._fill_obs_inplace(ref_next, info["last_action"])

        if self._use_iobinding:
            # 使用 IOBinding 进行零拷贝推理
            nn_action = self.nn_policy.infer_bound()
        else:
            nn_action = self.nn_policy.infer(self._obs_buf)

        motor_targets = self.nn2motor_action(nn_action)
        # 就地写入避免每步两次内存分配（last_action + motor_targets），
        # 同时不改变下游消费者的语义（读取 info["last_action"] 和 info["motor_targets"]）
        info["last_action"][...] = nn_action
        info["motor_targets"][...] = motor_targets
        info["nn_action"] = nn_action
        info["step"] += 1
        return motor_targets

    def get_nn_state(self, info, ref_state: np.ndarray, last_action: np.ndarray):
        """构建标准（非特权）观测向量

        观测向量组成（136 维）：
        - 本体感知：陀螺仪 (3) + 重力向量 (3)
        - 关节状态：关节位置误差 (29) + 关节速度 (29) + 上一帧动作 (29)
        - 参考运动：参考关节位置 (29) + 参考根节点高度 (1) + 参考重力向量 (3) + 参考根节点速度 (6)
        - 全局指令：偏航角 (cos/sin, 2) + XY 位置 (2)

        Args:
            info: 状态信息字典（包含陀螺仪、关节位置等）
            ref_state: 参考状态（下一帧），包含 qpos, kpt2gv_pose, kpt_cvel_in_gv
            last_action: 上一帧的动作，形状 (num_envs, 29)

        Returns:
            state: 观测向量，形状 (num_envs, 136)
        """
        ref_qpos = ref_state["qpos"][:, 7:]
        ref_qvel = ref_state["qvel"][:, 6:]

        gyro_pelvis = info["gyro_pelvis"]
        gvec_pelvis = info["gvec_pelvis"]
        joint_qpos = info["qpos"][:, self.qpos_ids_g1]
        joint_qvel = info["qvel"][:, self.dof_ids_g1]

        ref_root_gv_pose = ref_state["kpt2gv_pose"][:, 0]
        ref_root_cvel_in_gv = ref_state["kpt_cvel_in_gv"][:, 0]

        # 将偏航角编码为 [cos, sin] 以匹配训练逻辑
        yaw_cmd = np.stack([np.cos(info["yaw_d"]), np.sin(info["yaw_d"])], axis=-1)  # (num_envs, 2)
        # xy_d 已在导航坐标系中（R(-yaw_curr) @ (ref-curr)）；训练使用相同表示
        xy_cmd = info["xy_d"]

        state = np.hstack(
            [
                # 姿态状态
                gyro_pelvis,  # 3：骨盆陀螺仪（角速度）
                gvec_pelvis,  # 3：骨盆坐标系下的重力向量
                # 关节状态
                (joint_qpos - self._nom_jnt_qpos)[:, self.ctrl_id_obs],  # 29：关节位置误差（相对于默认站姿）
                joint_qvel[:, self.ctrl_id_obs],  # 29：关节速度  * scale_vel
                last_action,  # 29：上一帧动作（保证动作平滑）
                # 指令
                (ref_qpos - self._nom_jnt_qpos)[:, self.ctrl_id_obs],  # 29：参考关节位置（相对于默认站姿）
                # ref_qvel[:, self.ctrl_id_obs],  # 注释掉：参考关节速度（未使用）
                ref_root_gv_pose[:, 2, 3][:, None],  # 1：参考根节点高度（Z 坐标）
                -ref_root_gv_pose[:, :3, :3].transpose((0, 2, 1))[..., 2],  # 3：参考重力向量
                ref_root_cvel_in_gv,  # 6：参考根节点速度（线速度 + 角速度）
                # 全局
                yaw_cmd,  # (num_envs, 2)：偏航角指令 [cos(yaw), sin(yaw)]
                xy_cmd    # (num_envs, 2)：XY 位置指令（导航坐标系）
            ]
        ).astype(np.float32)
        return state

    def get_nn_priv_state(self, info, ref_state: np.ndarray, last_action: np.ndarray):
        """构建特权观测向量（包含额外的全局信息）

        特权模式使用仿真器中的额外信息（如关键点速度、相对位姿等），
        真机部署时不可用，仅用于训练和仿真评估。

        相比标准观测，额外包含：
        - 骨盆线速度 (3)
        - 实际根节点在重力坐标系下的线速度/角速度 (3+3)
        - 实际关键点在重力坐标系下的位姿 (14*6) 和速度 (14*6)
        - 下一帧参考与实际的残差：速度 (3)、关键点位姿 (14*6)、关键点速度 (14*6)

        Args:
            info: 状态信息字典
            ref_state: 参考状态（下一帧）
            last_action: 上一帧的动作

        Returns:
            privileged_state: 特权观测向量，维度大于 136
        """
        ref_qpos = ref_state["qpos"][:, 7:]
        ref_qvel = ref_state["qvel"][:, 6:]

        gyro_pelvis = info["gyro_pelvis"]
        gvec_pelvis = info["gvec_pelvis"]
        linvel_pelvis = info["linvel_pelvis"]
        joint_qpos = info["qpos"][:, self.qpos_ids_g1]
        joint_qvel = info["qvel"][:, self.dof_ids_g1]

        ref_root_gv_pose = ref_state["kpt2gv_pose"][:, 0]
        ref_root_cvel_in_gv = ref_state["kpt_cvel_in_gv"][:, 0]

        # 将偏航角编码为 [cos, sin] 以匹配训练的 priv_yaw_cmd
        priv_yaw_cmd = np.stack([np.cos(info["yaw_d"]), np.sin(info["yaw_d"])], axis=-1)  # (num_envs, 2)

        privileged_state = np.hstack(
            [
                # 姿态状态
                gyro_pelvis,  # 3：骨盆陀螺仪
                gvec_pelvis,  # 3：骨盆重力向量
                # 关节状态
                (joint_qpos - self._nom_jnt_qpos)[:, self.ctrl_id_obs],  # 29：关节位置误差
                joint_qvel[:, self.ctrl_id_obs],  # 29：关节速度
                last_action,  # 29：上一帧动作
                # 参考运动
                (ref_qpos - self._nom_jnt_qpos)[:, self.ctrl_id_obs],  # 29：参考关节位置
                # ref_qvel[:, self.ctrl_id_obs],  # 注释掉：参考关节速度
                # 提示状态（特权信息）
                linvel_pelvis,  # 3：骨盆线速度（局部坐标系）
                info["acu_root2gv_lin_vel"],  # 3：实际根节点在重力坐标系下的线速度
                info["acu_root2gv_ang_vel"],  # 3：实际根节点在重力坐标系下的角速度
                info["acu_kpt2gv_pose"][..., :3, :2].reshape(self.num_envs, -1),  # 14*6：实际关键点位姿（旋转矩阵前两列）
                info["acu_kpt2gv_pose"][..., :3, 3].reshape(self.num_envs, -1),  # 14*3：实际关键点位置
                info["acu_kpt_cvel_in_gv"].reshape(self.num_envs, -1),  # 14*6：实际关键点速度
                # 残差（下一帧参考 vs 当前实际）
                info["next_ref2acu_gv_vel"],  # 3：速度残差
                info["next_ref2acu_kpt_pose"][..., :3, :2].reshape(self.num_envs, -1),  # 14*6：关键点位姿残差
                info["next_ref2acu_kpt_pose"][..., :3, 3].reshape(self.num_envs, -1),  # 14*3：关键点位置残差
                info["next_ref2acu_kpt_cvel"].reshape(self.num_envs, -1),  # 14*6：关键点速度残差
                # 根节点指令
                ref_root_gv_pose[:, 2, 3][:, None],  # 1：参考根节点高度
                -ref_root_gv_pose[:, :3, :3].transpose((0, 2, 1))[..., 2],  # 3：参考重力向量
                ref_root_cvel_in_gv,  # 6：参考根节点速度
                # 全局
                priv_yaw_cmd,  # (num_envs, 2)：偏航角指令
                info["xy_d"],  # (num_envs, 2)：XY 位置指令
            ]
        ).astype(np.float32)
        return privileged_state

    def update_state(self, state: State, ref_state):
        """从 MuJoCo 仿真状态更新 info 字典

        提取传感器数据、计算坐标变换、更新关键点状态。
        支持两种模式：标准 MuJoCo（CPU）和 MJX（JAX GPU 加速）。

        Args:
            state: MuJoCo 仿真状态（包含 mj_data 或 mjx_data）
            ref_state: 参考状态字典（包含 ref_curr 和 ref_next）
        """
        if self.use_mjx:
            # MJX 模式：从 JAX 设备获取数据到 CPU
            mjx_data_np = jax.device_get(state.mjx_data)
            qpos = mjx_data_np.qpos.copy()
            qvel = mjx_data_np.qvel.copy()
            gyro_pelvis = mjx_sensor(self.mj_model, mjx_data_np, "gyro_pelvis")
            pelvis2world_rot = quat2mat(qpos[:, 3:7])
            pelvis2world_pos = qpos[:, :3]

            # 特权状态
            linvel_pelvis = mjx_sensor(
                self.mj_model, mjx_data_np, "local_linvel_pelvis"
            )
            torso2world_rot = mjx_data_np.site_xmat[:, self.site_id_torso_imu]
            feet_contact = self.get_mjx_feet_contact(mjx_data_np)
            kpt2wrd_rot = mjx_data_np.xmat[:, self.body_ids_kpt_full]
            kpt2wrd_pos = mjx_data_np.xpos[:, self.body_ids_kpt_full]
            acu_kpt_cvel_in_wrd = mjx_data_np.cvel[:, self.body_ids_kpt_full]

        else:
            # 标准 MuJoCo 模式
            mj_data = state.mj_data
            gyro_pelvis = mj_sensor(self.mj_model, mj_data, "gyro_pelvis")
            pelvis2world_rot = quat2mat(mj_data.qpos[3:7][None])
            pelvis2world_pos = mj_data.qpos[:3][None]
            qpos = mj_data.qpos.copy()[None]
            qvel = mj_data.qvel.copy()[None]

            # 特权状态
            linvel_pelvis = mj_sensor(self.mj_model, mj_data, "local_linvel_pelvis")
            torso2world_rot = mj_data.site_xmat[self.site_id_torso_imu]
            torso2world_rot = torso2world_rot.reshape(1, 3, 3)
            feet_contact = self.get_mj_feet_contact(mj_data)[None]
            kpt2wrd_rot = mj_data.xmat[self.body_ids_kpt_full].reshape(-1, 3, 3)
            kpt2wrd_pos = mj_data.xpos[self.body_ids_kpt_full][None]
            acu_kpt_cvel_in_wrd = mj_data.cvel[self.body_ids_kpt_full][None]

        # 更新基本状态
        gvec_pelvis = -pelvis2world_rot.transpose(0, 2, 1)[..., 2]
        self.info["gyro_pelvis"][:] = gyro_pelvis.copy()
        self.info["linvel_pelvis"][:] = linvel_pelvis.copy()
        self.info["gvec_pelvis"][:] = gvec_pelvis.copy()
        self.info["qpos"][:] = qpos.copy()
        self.info["qvel"][:] = qvel.copy()

        # 计算导航坐标系（navigation frame）
        # 导航坐标系：Z 轴与重力对齐，只保留偏航角，原点投影到地面
        navi2world_rot = batch_base2navi(pelvis2world_rot)
        pelvis2navi_rot = navi2world_rot.transpose(0, 2, 1) @ pelvis2world_rot
        torso2navi_rot = navi2world_rot.transpose(0, 2, 1) @ torso2world_rot
        self.info["navi2world_rot"] = navi2world_rot.copy()
        self.info["navi_pelvis_rpy"][:] = (
            R.from_matrix(pelvis2navi_rot).as_euler("xyz").copy()
        )
        self.info["navi_torso_rpy"][:] = (
            R.from_matrix(torso2navi_rot).as_euler("xyz").copy()
        )
        self.info["feet_contact"][:] = feet_contact.copy()

        # 计算重力坐标系（gravity view frame）
        # 原点位于骨盆 XY 位置，Z 轴与重力对齐
        acu_gv2wrd_pose = np.full((self.num_envs, 4, 4), np.eye(4))
        acu_gv2wrd_pose[:, :3, :3] = navi2world_rot
        acu_gv2wrd_pose[:, :2, 3] = pelvis2world_pos[:, :2]
        # 关键点在世界坐标系下的位姿
        acu_kpt2wrd_pose = np.full((self.num_envs, self.num_kpt, 4, 4), np.eye(4))
        acu_kpt2wrd_pose[:, :, :3, :3] = kpt2wrd_rot
        acu_kpt2wrd_pose[:, :, :3, 3] = kpt2wrd_pos
        # 关键点在重力坐标系下的位姿：inv(gv2wrd) @ kpt2wrd
        acu_kpt2gv_pose = np.linalg.inv(acu_gv2wrd_pose[:, None]) @ acu_kpt2wrd_pose
        self.info["acu_kpt2gv_pose"] = acu_kpt2gv_pose

        # 关键点速度从世界坐标系转换到重力坐标系
        acu_kpt_cvel_in_gv = np.zeros_like(acu_kpt_cvel_in_wrd)
        R_wrd2gv = np.swapaxes(acu_gv2wrd_pose[..., :3, :3], -1, -2)  # R_gv2wrd^T = R_wrd2gv
        acu_kpt_cvel_in_gv[..., :3] = np.einsum("...ij,...kj->...ki", R_wrd2gv, acu_kpt_cvel_in_wrd[..., :3])
        acu_kpt_cvel_in_gv[..., 3:] = np.einsum("...ij,...kj->...ki", R_wrd2gv, acu_kpt_cvel_in_wrd[..., 3:])
        self.info["acu_kpt_cvel_in_gv"][:] = acu_kpt_cvel_in_gv

        # 更新坐标指令（偏航角和 XY 位置）
        self.update_coord_cmd(ref_state)

        # 如果存在下一帧参考，计算特权信息（残差）
        if "ref_next" in ref_state:
            ref_next = ref_state["ref_next"]

            next_ref_gv2wrd_pose = ref_next["gv2wrd_pose"]
            self.info["next_ref2acu_gv_pose"] = (
                np.linalg.inv(acu_gv2wrd_pose) @ next_ref_gv2wrd_pose
            )
            next_ref2acu_kpt_pose = (
                np.linalg.inv(acu_kpt2gv_pose) @ ref_next["kpt2gv_pose"]
            )
            self.info["next_ref2acu_kpt_pose"] = next_ref2acu_kpt_pose

            ref_next_navi_vel = ref_next["gv_vel"]
            acu_root2gv_lin_vel = pelvis2navi_rot @ linvel_pelvis
            acu_root2gv_ang_vel = pelvis2navi_rot @ gyro_pelvis
            self.info["acu_root2gv_lin_vel"] = acu_root2gv_lin_vel
            self.info["acu_root2gv_ang_vel"] = acu_root2gv_ang_vel
            # self.info["acu_navi_vel"] = np.hstack(
            #     [navi_pelvis_lin_vel[:, :2], navi_pelvis_ang_vel[:, 2:3]]
            # )
            # 计算速度残差：参考速度 - 实际速度
            next_vel_lin_res = ref_next_navi_vel[:, :2] - acu_root2gv_lin_vel[:, :2]
            next_vel_ang_res = ref_next_navi_vel[:, 2] - acu_root2gv_ang_vel[:, 2]
            self.info["next_ref2acu_gv_vel"] = np.hstack(
                [next_vel_lin_res, next_vel_ang_res[:, None]]
            )
            # self.info["ref_next_navi_vel"][:] = ref_next["navi_vel"]
            # self.info["ref_next_kpt_npose"] = ref_next["kpt_cvel_in_gv"]
            # self.info["ref_next_kpt_cvel"] = ref_next["kpt_cvel"]
            # self.info["next_jnt_qpos_res"] = ref_next["qpos"][:, 7:] - qpos[:, 7:]
            # self.info["next_jnt_qvel_res"] = ref_next["qvel"][:, 6:] - qvel[:, 6:]
            # 关键点速度残差
            self.info["next_ref2acu_kpt_cvel"] = (
                ref_next["kpt_cvel_in_gv"] - acu_kpt_cvel_in_gv
            )

    def update_coord_cmd(self, ref_state):
        """更新坐标指令：计算目标偏航角和 XY 位置

        将参考轨迹的全局位置转换为相对于当前机器人状态的局部指令。
        这是实现"导航坐标系"控制的关键步骤。

        Args:
            ref_state: 参考状态字典，包含 ref_curr（当前帧参考）
        """
        curr_ref_state = ref_state["ref_curr"]
        q_ref = curr_ref_state["qpos"][:, 3:7]
        q_curr = self.info["qpos"][:, 3:7]
        yaw_ref = quat2yaw(q_ref)
        yaw_curr = quat2yaw(q_curr)
        yaw_cmd = curr_ref_state.get("yaw_cmd", np.zeros_like(yaw_curr))
        yaw_target = yaw_cmd + yaw_ref
        # 计算偏航角误差，并包装到 [-pi, pi]
        yaw_d = WarpPi(yaw_target - yaw_curr)

        p_ref = curr_ref_state["qpos"][:, :2]
        p_curr = self.info["qpos"][:, :2]

        # 构建旋转矩阵 R(yaw_cmd)，用于将 XY 指令旋转到机器人坐标系
        c, s = np.cos(yaw_cmd), np.sin(yaw_cmd)
        R_m = np.stack([
            np.stack([c, -s], axis=-1),
            np.stack([s, c], axis=-1),
        ], axis=-2)
        xy_ref = p_ref
        xy_curr = p_curr
        xy_cmd = curr_ref_state.get("xy_cmd", np.zeros_like(xy_curr))
        # 计算目标 XY 位置（世界坐标系）
        xy_target = np.einsum("...ij,...j->...i", R_m, (xy_cmd + xy_ref))
        xy_d = xy_target - xy_curr

        # 将 XY 误差转换到机器人局部坐标系（去除当前偏航角的影响）
        c, s = np.cos(-yaw_curr), np.sin(-yaw_curr)
        R_m = np.stack([
            np.stack([c, -s], axis=-1),
            np.stack([s, c], axis=-1),
        ], axis=-2)
        xy_d = np.einsum("...ij,...j->...i", R_m, xy_d)
        # xy_d = np.clip(xy_d, self.coord_cfg["xy_move_range"][0], self.coord_cfg["xy_move_range"][1])

        self.info["yaw_d"] = yaw_d
        self.info["xy_d"] = xy_d

    def get_mj_feet_contact(self, mj_data: mujoco.MjData) -> np.ndarray:
        """获取左右脚的接触状态（标准 MuJoCo 模式）

        接触状态编码：
            -1: 完全在空中（无接触）
             1: 部分接触（部分或全部脚部几何体接触）

        Args:
            mj_data: MuJoCo 数据

        Returns:
            接触状态数组，形状 (2,)，[左脚, 右脚]
        """
        left_contacts = np.array(
            [
                mj_coll(mj_data, geom_id, self.geom_id_floor)
                for geom_id in self.geom_ids_left_feet
            ]
        )
        right_contacts = np.array(
            [
                mj_coll(mj_data, geom_id, self.geom_id_floor)
                for geom_id in self.geom_ids_right_feet
            ]
        )

        left_state = np.where(left_contacts.any(), 1, -1)
        right_state = np.where(right_contacts.any(), 1, -1)

        return np.array([left_state, right_state])

    def get_mjx_feet_contact(self, mjx_data: mjx.Data):
        """获取左右脚的接触状态（MJX GPU 加速模式）

        Args:
            mjx_data: MJX 数据（JAX 数组）

        Returns:
            接触状态数组，形状 (num_envs, 2)
        """
        left_contacts = np.array(
            [
                mjx_coll(mjx_data, geom_id, self.geom_id_floor)
                for geom_id in self.geom_ids_left_feet
            ]
        )
        right_contacts = np.array(
            [
                mjx_coll(mjx_data, geom_id, self.geom_id_floor)
                for geom_id in self.geom_ids_right_feet
            ]
        )

        left_state = np.where(left_contacts.any(axis=0), 1, -1)[:, None]
        right_state = np.where(right_contacts.any(axis=0), 1, -1)[:, None]

        return np.concatenate([left_state, right_state], axis=1)

    def nn2motor_action(self, nn_action):
        """将神经网络输出转换为电机目标位置

        动作空间是残差形式：motor_target = default_qpos + nn_action * action_scale
        这种设计使得策略网络学习"相对于默认站姿的偏移"，而不是绝对关节位置。

        Args:
            nn_action: 神经网络输出，形状 (num_envs, 29)

        Returns:
            motor_targets: 电机目标位置，形状 (num_envs, 29)
        """
        motor_targets = self._nom_jnt_qpos.copy()
        motor_targets[:, self.ctrl_id_act] = (
            self._nom_jnt_qpos[:, self.ctrl_id_act]
            + nn_action
            * self.env_config.action_scale
            * self.act_scale[self.ctrl_id_act]
        )
        return motor_targets

