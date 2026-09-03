"""带球拍的网球场景加载：在 scene_tennis.xml 基础上把球拍动态挂到右手腕。

用法：
    from utils.tennis_scene import load_tennis_model
    mj_model = load_tennis_model(racket_mass=0.3)

球拍是 right_wrist_yaw_link 下的无关节刚体（参考 noitom-robotics/AdaPT
g1_rightracket.xml 的挂接方式），机器人 nq 保持 36；场景中的网球 freejoint
追加在 qpos 尾部（qposadr=36），机器人侧的索引不受影响。

网球物理参数参考 wfzark/tennis-robot-rl：半径 0.0335 m，质量 0.0577 kg。
"""

from __future__ import annotations

import mujoco
import numpy as np

from tracking import constants as consts

TENNIS_SCENE_XML = consts.ROOT_PATH / "scene_tennis.xml"

# 拍面法向从腕部坐标系 z 轴转 90°（圆柱默认沿 z，转后拍面法向为 y）
_RACKET_FACE_QUAT = (0.7071068, 0.0, 0.7071068, 0.0)


def attach_racket(spec: mujoco.MjSpec, mass: float = 0.3) -> None:
    """在 right_wrist_yaw_link 下挂接无关节球拍 body（修改 spec，原地生效）。"""
    wrist = spec.body("right_wrist_yaw_link")
    racket = wrist.add_body(name="racket", pos=[0.05, 0, 0])
    # 拍柄 + 拍杆
    racket.add_geom(
        name="racket_shaft", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0.08, 0, 0, 0.30, 0, 0], size=[0.012, 0, 0],
        rgba=[0.15, 0.15, 0.18, 1],
    )
    # 拍面（视觉）
    racket.add_geom(
        name="racket_frame", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.17, 0.012, 0.012], pos=[0.45, 0, 0],
        quat=list(_RACKET_FACE_QUAT), rgba=[0.15, 0.15, 0.18, 1],
    )
    # 拍面（碰撞代理：薄圆柱）
    col = racket.add_geom(
        name="racket_collision", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.17, 0.012, 0.012], pos=[0.45, 0, 0],
        quat=list(_RACKET_FACE_QUAT), rgba=[0.2, 0.6, 0.2, 0.3],
    )
    col.condim = 3
    col.friction = [0.5, 0.005, 0.0001]
    col.solref = [0.005, 1]
    # 拍面中心（用于命中检测/奖励）
    racket.add_site(name="racket_point", pos=[0.45, 0, 0], size=[0.005, 0.005, 0.005])

    racket.explicitinertial = True
    racket.ipos = [0.35, 0, 0]
    racket.mass = mass
    racket.fullinertia = [0.003, 0.003, 0.0002, 0, 0, 0]

    # 拍 - 球接触对（XML 中无法静态声明，因为球拍是动态挂接的）
    spec.add_pair(
        name="racket_ball", geomname1="racket_collision", geomname2="tennis_ball_geom",
        condim=3,
    )


def load_tennis_model(racket_mass: float = 0.3,
                      xml_path: str | None = None) -> mujoco.MjModel:
    """加载带球拍+网球+球场的场景模型。

    Args:
        racket_mass: 球拍质量（kg）。跟踪策略训练时手上无质量，
            0.3 kg 为轻量拍，用于先打通链路；加重会显著影响挥拍跟踪。
        xml_path: 场景 XML 路径，默认 scene_tennis.xml。
    """
    spec = mujoco.MjSpec.from_file(xml_path or str(TENNIS_SCENE_XML))
    attach_racket(spec, mass=racket_mass)
    mj_model = spec.compile()
    return mj_model


def get_ball_qposadr(mj_model: mujoco.MjModel) -> int:
    """网球 freejoint 在 qpos 中的起始索引（机器人 qpos 36 维之后）。"""
    return int(mj_model.joint("tennis_ball_free").qposadr)


def reset_ball(mj_data: mujoco.MjData, mj_model: mujoco.MjModel,
               pos=(2.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0)) -> None:
    """把网球放到指定位姿并设置速度（世界系）。"""
    adr = get_ball_qposadr(mj_model)
    dof = int(mj_model.joint("tennis_ball_free").dofadr[0])
    mj_data.qpos[adr:adr + 3] = pos
    # 四元数保持单位
    mj_data.qpos[adr + 3:adr + 7] = (1, 0, 0, 0)
    mj_data.qvel[dof:dof + 3] = vel


def get_ball_state(mj_data: mujoco.MjData, mj_model: mujoco.MjModel):
    """读取网球世界系位置与速度。"""
    adr = get_ball_qposadr(mj_model)
    dof = int(mj_model.joint("tennis_ball_free").dofadr[0])
    return mj_data.qpos[adr:adr + 3].copy(), mj_data.qvel[dof:dof + 3].copy()
