from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

from utils.hand import load_mj_model
from utils.sim_base import BaseSim, State


def get_qpos_ids(mj_model, names):
    return np.hstack([mj_model.joint(n).qposadr for n in names])


def get_dof_ids(mj_model, names):
    return np.hstack([mj_model.joint(n).dofadr for n in names])


def get_sensor_data(mj_model, mj_data, sensor_name: str) -> np.ndarray:
    """Gets sensor data given sensor name."""
    sensor_id = mj_model.sensor(sensor_name).id
    sensor_adr = mj_model.sensor_adr[sensor_id]
    sensor_dim = mj_model.sensor_dim[sensor_id]
    return mj_data.sensordata[sensor_adr : sensor_adr + sensor_dim]


def get_collision_info(
    contact: Any, geom1: int, geom2: int
) -> tuple[np.ndarray, np.ndarray]:
    """Get the distance and normal of the collision between two geoms."""
    mask = (np.array([geom1, geom2]) == contact.geom).all(axis=1)
    mask |= (np.array([geom2, geom1]) == contact.geom).all(axis=1)
    idx = np.where(mask, contact.dist, 1e4).argmin()
    dist = contact.dist[idx] * mask[idx]
    normal = (dist < 0) * contact.frame[idx, :3]
    return dist, normal


def geoms_colliding(state: mujoco.MjData, geom1: int, geom2: int):
    """Return True if the two geoms are colliding."""
    if len(state.contact) == 0:
        return 0
    return get_collision_info(state.contact, geom1, geom2)[0] < 0


class MJSim(BaseSim):
    """MuJoCo 仿真器封装（CPU 版本）

    提供标准的物理仿真接口，包括：
    - PD 控制器：将目标关节位置转换为力矩
    - 物理步进：调用 mujoco.mj_step() 进行物理仿真
    - 可视化：支持被动查看器（passive viewer）

    Attributes:
        kps: 位置环 PD 增益（比例增益）
        kds: 速度环 PD 增益（微分增益）
        torque_limit: 各关节的力矩限制
        init_qpos: 初始关节位置
    """

    kps: np.ndarray
    kds: np.ndarray
    torque_limit: np.ndarray
    init_qpos: np.ndarray

    def __init__(self, xml_path: str, ctrl_dt=0.02, sim_dt=0.001, headless=False):
        """初始化 MuJoCo 仿真器

        Args:
            xml_path: MuJoCo XML 模型文件路径
            ctrl_dt: 控制周期（秒），默认 0.02s（50Hz）
            sim_dt: 物理仿真步长（秒），默认 0.001s（1ms）
                   每个控制周期内仿真 ctrl_dt / sim_dt = 20 步
            headless: 是否无头模式（不启动可视化窗口）
        """
        self.ctrl_dt = ctrl_dt
        self.sim_dt = sim_dt
        self.num_sim_substeps = int(self.ctrl_dt / self.sim_dt)
        # 手部感知加载：安装 HAND 末端执行器（刚性连接，nq 保持 36）
        self.mj_model = load_mj_model(xml_path)
        self.mj_model.opt.timestep = sim_dt
        self.headless = headless

    def init_state(self) -> State:
        """初始化仿真状态

        Returns:
            State: 包含 mj_data 的状态对象
        """
        mj_data = mujoco.MjData(self.mj_model)

        if not self.headless:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, mj_data)
        return State(mj_data=mj_data)

    def reset(self, state: State, sim_ids=None) -> State:
        """重置仿真状态到初始位置

        Args:
            state: 当前状态
            sim_ids: 未使用（为了与 MJX 版本接口一致）

        Returns:
            State: 重置后的状态
        """
        mj_data = state.mj_data

        mj_data.qpos[:] = self.init_qpos
        mj_data.qvel[:] = 0.0
        mj_data.ctrl[:] = 0.0
        mujoco.mj_forward(self.mj_model, mj_data)
        return State(mj_data=mj_data)

    def step(self, state: State, action: np.ndarray) -> State:
        """执行一步仿真（一个控制周期）

        PD 控制律：
            tau = kp * (q_target - q) - kd * qd
        其中：
            - q_target: 目标关节位置（action）
            - q: 当前关节位置
            - qd: 当前关节速度
            - kp, kd: PD 增益

        Args:
            state: 当前状态
            action: 目标关节位置，形状 (29,)

        Returns:
            State: 更新后的状态
        """
        mj_data = state.mj_data

        # 网球场景中 qpos/qvel 尾部含网球 freejoint，PD 只作用于机器人关节
        nq = self.mj_model.nq
        nv = self.mj_model.nv
        if nq > 36:
            qpos_jnt = mj_data.qpos[7:36]
            qvel_jnt = mj_data.qvel[6:35]
        else:
            qpos_jnt = mj_data.qpos[7:]
            qvel_jnt = mj_data.qvel[6:]

        for _ in range(self.num_sim_substeps):
            # 计算 PD 控制力矩
            torques = self.kps * (action - qpos_jnt) + self.kds * (-qvel_jnt)
            # 限幅：防止力矩超过电机极限
            mj_data.ctrl[:] = np.clip(torques, -self.torque_limit, self.torque_limit)
            # 物理仿真步进
            mujoco.mj_step(self.mj_model, mj_data)

        return State(mj_data=mj_data)

    def view(self, state: State, sim_id: int = 0) -> None:
        """更新可视化（仅在非无头模式下有效）

        Args:
            state: 当前状态
            sim_id: 仿真 ID（未使用，为了与 MJX 版本接口一致）
        """
        if not self.headless:
            self.viewer.sync()
        else:
            print(
                "Headless mode enabled: no graphical viewer is available for rendering."
            )
