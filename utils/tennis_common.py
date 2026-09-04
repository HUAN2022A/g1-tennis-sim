"""网球场景仿真的公共工具：build_sim / site_state / contact_event。

从 scripts/tennis_hit_test.py 抽出，供命中测试（#6）与 V0 状态机（#8）
共用，避免复制。scene_tennis.xml + utils/tennis_scene.py 动态挂拍，
nq=43（机器人 36 + 网球 freejoint 7）。
"""

from __future__ import annotations

import numpy as np
import mujoco

from tracking import constants as consts
from tracking.infer_utils import G1TrackMjSim
from utils.tennis_scene import load_tennis_model, get_ball_qposadr


def build_tennis_sim(racket_mass: float = 0.05, ctrl_dt: float = 0.02,
                     sim_dt: float = 0.001, margin: float = 0.0,
                     headless: bool = True) -> tuple[G1TrackMjSim, int, int, int]:
    """构建网球场景仿真（nq=43）。

    Returns:
        (sim, site_id, ball_qposadr, ball_dofadr)
    """
    sim = G1TrackMjSim(
        init_qpos=consts.DEFAULT_QPOS.copy(),
        headless=headless,
        ctrl_dt=ctrl_dt,
        sim_dt=sim_dt,
        xml_path=str(consts.ROOT_PATH / "scene_tennis.xml"),
    )
    sim.mj_model = load_tennis_model(racket_mass=racket_mass)
    sim.mj_model.opt.timestep = sim_dt
    if margin > 0:
        rid = sim.mj_model.geom("racket_collision").id
        bid = sim.mj_model.geom("tennis_ball_geom").id
        sim.mj_model.geom_margin[rid] = margin
        sim.mj_model.geom_margin[bid] = margin
    full_init = np.zeros(sim.mj_model.nq)
    full_init[:36] = sim.init_qpos
    ball_adr = get_ball_qposadr(sim.mj_model)
    full_init[ball_adr + 3] = 1.0  # 球四元数 w=1
    sim.init_qpos = full_init

    site_id = sim.mj_model.site("racket_point").id
    ball_dof = int(sim.mj_model.joint("tennis_ball_free").dofadr[0])
    return sim, site_id, ball_adr, ball_dof


def set_ball(mj_data, ball_adr: int, ball_dof: int, pos, vel=(0.0, 0.0, 0.0)) -> None:
    """设置网球位姿与速度（世界系，四元数归单位）。"""
    mj_data.qpos[ball_adr:ball_adr + 3] = pos
    mj_data.qpos[ball_adr + 3:ball_adr + 7] = (1, 0, 0, 0)
    mj_data.qvel[ball_dof:ball_dof + 3] = vel
    mj_data.qvel[ball_dof + 3:ball_dof + 6] = 0.0


def get_ball(mj_data, ball_adr: int, ball_dof: int) -> tuple[np.ndarray, np.ndarray]:
    """读取网球世界系位置与速度。"""
    return (mj_data.qpos[ball_adr:ball_adr + 3].copy(),
            mj_data.qvel[ball_dof:ball_dof + 3].copy())


def site_state(mj_model, mj_data, site_id: int):
    """racket_point site 的世界位置、拍面法向（site 系 y 轴）、世界系线速度。"""
    pos = mj_data.site_xpos[site_id].copy()
    xmat = mj_data.site_xmat[site_id].reshape(3, 3)
    normal = xmat[:, 1].copy()  # 拍面法向（见 utils/tennis_scene.py 挂拍注释）
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(
        mj_model, mj_data, mujoco.mjtObj.mjOBJ_SITE, site_id, vel, 0
    )
    return pos, normal, vel[3:].copy()


def contact_event(mj_model, mj_data) -> bool:
    """当前控制步是否有拍面-球接触。"""
    rid = mj_model.geom("racket_collision").id
    bid = mj_model.geom("tennis_ball_geom").id
    for i in range(mj_data.ncon):
        g1, g2 = mj_data.contact[i].geom1, mj_data.contact[i].geom2
        if {g1, g2} == {rid, bid}:
            return True
    return False
