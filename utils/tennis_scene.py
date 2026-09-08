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

# 拍面法向沿腕部坐标系 z 轴（圆柱默认沿 z，无需旋转，quat=identity），
# 拍面平面由拍柄方向 x 与腕部 y 张成。
# V0 起这里是 (0.7071,0,0.7071,0)（绕 y 转 90°，法向落到拍柄方向 x，
# 拍面平面⊥拍柄，注释却写"转后法向为 y"）——2026-09-08 修正。
# 法向选 z 是 2026-09-08 实测选定（logs/racketfix_candidates.log，
# 8 球干净快测）：候选 ±y/±z 四种里 ±z 命中 3/8 且接触面内半径
# 0.098-0.143（拍面中部）；±y 仅 1/8 且接触在框缘 r≈0.175；
# 旧 ±x 2/8、接触贴外缘带 r≈0.15。圆柱轴向翻转对称，±z 物理等价。
_RACKET_FACE_QUAT = (1.0, 0.0, 0.0, 0.0)


def attach_racket(spec: mujoco.MjSpec, mass: float = 0.3) -> None:
    """在 right_wrist_yaw_link 下挂接无关节球拍 body（修改 spec，原地生效）。"""
    wrist = spec.body("right_wrist_yaw_link")
    racket = wrist.add_body(name="racket", pos=[0.05, 0, 0])
    # 拍柄（握把）
    grip = racket.add_geom(
        name="racket_grip", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0.03, 0, 0, 0.18, 0, 0], size=[0.016, 0, 0],
        rgba=[0.13, 0.13, 0.16, 1],
    )
    grip.contype = 0
    grip.conaffinity = 0
    # 拍喉（握把与拍头之间；终点与拍头环近端 0.28 衔接，收到 0.30 略搭接）
    throat = racket.add_geom(
        name="racket_throat", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0.18, 0, 0, 0.30, 0, 0], size=[0.009, 0, 0],
        rgba=[0.16, 0.16, 0.19, 1],
    )
    throat.contype = 0
    throat.conaffinity = 0
    # 拍头椭圆框（细胶囊围成椭圆环，纯视觉）。拍头平面含拍柄轴（x）：
    # 长轴沿 x（半轴 a=0.17，拍头从 0.28 延伸到 0.62），短轴沿 y（半轴 b），
    # 法向沿 z——与真实网球拍一致（拍头与拍柄同面延伸），且与碰撞盘
    # （法向 z）同面。
    # b 取 0.15：碰撞盘半径 0.17，b=0.15 把"接触点落在画出的弦床之外"的
    # 边带压到 2 cm（b=0.13 边带 4 cm；圆形拍头 b=0.17 则完全无带）。
    cx = 0.45
    a, b = 0.17, 0.15  # 半长轴(X) 半短轴(Y)
    n_seg = 14
    for i in range(n_seg):
        t0 = 2 * np.pi * i / n_seg
        t1 = 2 * np.pi * (i + 1) / n_seg
        p0 = [cx + a * np.cos(t0), b * np.sin(t0), 0]
        p1 = [cx + a * np.cos(t1), b * np.sin(t1), 0]
        seg = racket.add_geom(
            name=f"racket_frame_{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[*p0, *p1], size=[0.010, 0, 0],
            rgba=[0.16, 0.16, 0.19, 1],
        )
        seg.contype = 0
        seg.conaffinity = 0
    # 拍面弦床（半透明薄盒，纯视觉）：平面内 x-y，薄方向 z（=拍面法向）
    bed = racket.add_geom(
        name="racket_strings", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[a * 0.95, b * 0.95, 0.012], pos=[cx, 0, 0],
        rgba=[0.92, 0.92, 0.92, 0.22],
    )
    bed.contype = 0
    bed.conaffinity = 0
    # 拍面（碰撞代理：薄圆柱）
    col = racket.add_geom(
        name="racket_collision", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.17, 0.012, 0.012], pos=[0.45, 0, 0],
        quat=list(_RACKET_FACE_QUAT), rgba=[0.2, 0.6, 0.2, 0.3],
    )
    col.condim = 3
    col.friction = [0.5, 0.005, 0.0001]
    col.solref = [0.005, 1]
    # 拍面中心（用于命中检测/奖励）。quat 绕 x 转 +90°：使 site 系 y 轴对准
    # 腕系 z 轴（=拍面法向，见 _RACKET_FACE_QUAT 注释）——这样
    # utils/tennis_common.py site_state 取 site 系 y 轴做法向的约定严格成立。
    # 只改朝向，位置不变，命中检测（site_xpos）不受影响。
    racket.add_site(name="racket_point", pos=[0.45, 0, 0],
                    quat=[0.7071068, 0.7071068, 0.0, 0.0],
                    size=[0.005, 0.005, 0.005])

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
