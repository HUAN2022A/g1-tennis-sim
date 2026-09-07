"""网球发球 / 球物理域随机化 / 阻力助手（自 scripts/tennis_v1_robust.py 抽取）。

供 V1/V2/V3 共用：发射条件随机化（可行域采样）、带阻力发球速度设计、
球物理 DR 施加/恢复、二次空气阻力外力、带阻力的控制步。
"""

from __future__ import annotations

import numpy as np
import mujoco

from utils.ball_predict import predict_ball_trajectory

G = 9.81


def design_serve(p_swing: np.ndarray, flight_time: float,
                 serve_pos: tuple = ()) -> tuple[np.ndarray, np.ndarray]:
    """设计发射位置/速度：抛体解使球在 flight_time 后到达 P_swing。

    默认发射点在 P_swing 上游 +x 侧（场地对面方向），高度 +1m。
    v = (P - p0 + 0.5*g*T^2*e_z) / T
    """
    if serve_pos:
        p0 = np.asarray(serve_pos, dtype=np.float64)
    else:
        p0 = p_swing + np.array([2.8, 0.0, 1.0])
    T = flight_time
    v = (p_swing - p0 + np.array([0.0, 0.0, 0.5 * G * T * T])) / T
    return p0, v


def design_serve_with_drag(target: np.ndarray, T: float, k: float,
                           serve_height: float, serve_back: float
                           ) -> tuple[np.ndarray, np.ndarray]:
    """带阻力的发射速度设计：迭代求解使球在 T 秒后到达 target。

    发射器使用已辨识的球模型（k_pred，与预测侧一致）：名义弹道解出发射
    速度后，用带阻力积分器模拟到达点，按误差迭代修正（2-3 次收敛到 <1mm）。
    真实球（k_true ≠ k_pred）的残差是 DR 的一部分。
    """
    p0 = target + np.array([serve_back, 0.0, serve_height])
    v = (target - p0 + np.array([0.0, 0.0, 0.5 * G * T * T])) / T
    for _ in range(4):
        traj, ts = predict_ball_trajectory(p0, v, k=k, dt=0.005, horizon=T + 0.2)
        # 找 t=T 时刻位置（线性插值）
        i = int(T / 0.005)
        frac = (T - ts[i - 1]) / 0.005
        p_T = traj[i - 1] + frac * (traj[i] - traj[i - 1])
        err = target - p_T
        if np.linalg.norm(err) < 1e-3:
            break
        v = v + err / T  # 一阶修正：速度按位置误差/T 调整
    return p0, v


def sample_serve(p_swing: np.ndarray, rng: np.random.Generator,
                 feas_radius: float, flight_lo: float, flight_hi: float,
                 serve_height: float, serve_back: float, design_k: float = 0.0
                 ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """在可行域内随机采样发射条件。

    拦截目标 = P_swing + 圆盘采样偏移（水平面内随机方向 + 小幅 z 偏移）。
    飞行时间先采 U(lo, hi)，再量化到控制帧网格（T = t_lead + round((T-
    t_lead)/dt)*dt）——发射器在 50Hz 控制网格上调度发射；不量化时理想的
    触发时刻落在半帧边界上（50Hz 下 1 帧 = 14 m/s 挥速 × 0.02s = 0.28m
    拍面位移），触发帧的取整方向随机、系统性地砍掉一半命中率。
    发射速度按 design_k（已辨识球模型，与预测侧一致）的带阻力弹道解出。

    返回 (serve_pos, serve_vel, T_flight, target_point)。
    """
    # 圆盘采样：极径 sqrt 均匀（盘上均匀），方向均匀。
    # z 偏移收窄到 ±0.02：挥拍峰值附近拍面 z 向速度 ~5 m/s（0.1 m/帧），
    # z 向偏移对命中几何的杀伤远大于水平偏移，接近随机化上限。
    r = feas_radius * np.sqrt(rng.random())
    theta = rng.random() * 2 * np.pi
    offset = np.array([r * np.cos(theta), r * np.sin(theta),
                       rng.uniform(-0.02, 0.02)])
    target = p_swing + offset
    T = rng.uniform(flight_lo, flight_hi)
    # 帧网格量化（ctrl_dt=0.02）：飞行时间对齐 t_lead + 整数帧，
    # 使球到达时刻与挥拍峰值帧同相
    grid = 0.02
    T = 0.26 + round((T - 0.26) / grid) * grid
    T = float(np.clip(T, flight_lo - grid, flight_hi + grid))
    if design_k > 0:
        p0, v = design_serve_with_drag(target, T, design_k,
                                       serve_height, serve_back)
    else:
        p0 = target + np.array([serve_back, 0.0, serve_height])
        v = (target - p0 + np.array([0.0, 0.0, 0.5 * G * T * T])) / T
    return p0, v, T, target


def apply_ball_dr(mj_model, rng: np.random.Generator,
                  k_lo: float = 0.005, k_hi: float = 0.02) -> dict:
    """球物理域随机化（LATENT Table 2 球相关项）。返回采样参数记录。

    - 质量 ×U(0.95, 1.05)：改 body_mass
    - 切向摩擦 U(0.4, 0.7)：改 geom_friction[0]（滑动摩擦系数）
    - 地面弹性 U(0.71, 0.79)：MuJoCo 中反弹由接触 solref/solimp 控制，
      这里按比例调制球 geom 的 solref 阻尼（restitution 越小反弹越弱）
    - 空气阻力 k U(k_lo, k_hi)：不改模型，由外力每子步施加（见 apply_drag）
    """
    bid = mj_model.body("tennis_ball").id
    gid = mj_model.geom("tennis_ball_geom").id
    mass_scale = rng.uniform(0.95, 1.05)
    fric = rng.uniform(0.4, 0.7)
    restitution = rng.uniform(0.71, 0.79)
    k_drag = rng.uniform(k_lo, k_hi)

    mj_model.body_mass[bid] *= mass_scale
    # 质量变化后需重算惯量（球惯量 ∝ m r^2）
    mj_model.body_inertia[bid] *= mass_scale
    mj_model.geom_friction[gid][0] = fric
    # 反弹强度：solref[1]（damping ratio）越大耗散越多、反弹越弱。
    # restitution 0.75（名义球）≈ 现有 solref；按 (restitution/0.75)^2 缩放
    # 耗散项，使反弹高度近似按弹性系数平方变化。
    base_damping = 0.2  # scene_tennis.xml solref="0.01 0.2"
    mj_model.geom_solref[gid][1] = base_damping * (0.75 / restitution) ** 2

    return {"mass_scale": float(mass_scale), "friction": float(fric),
            "restitution": float(restitution), "k_drag": float(k_drag)}


def restore_ball_physics(mj_model, base_state: dict) -> None:
    """恢复球的标称物理参数（DR 是逐球采样的，下球前恢复再重采）。"""
    bid = mj_model.body("tennis_ball").id
    gid = mj_model.geom("tennis_ball_geom").id
    mj_model.body_mass[bid] = base_state["mass"]
    mj_model.body_inertia[bid] = base_state["inertia"]
    mj_model.geom_friction[gid] = base_state["friction"]
    mj_model.geom_solref[gid] = base_state["solref"]


def snapshot_ball_physics(mj_model) -> dict:
    bid = mj_model.body("tennis_ball").id
    gid = mj_model.geom("tennis_ball_geom").id
    return {"mass": mj_model.body_mass[bid].copy(),
            "inertia": mj_model.body_inertia[bid].copy(),
            "friction": mj_model.geom_friction[gid].copy(),
            "solref": mj_model.geom_solref[gid].copy()}


def apply_drag(mj_data, ball_dof: int, ball_body_id: int,
               mj_model, k: float) -> None:
    """二次空气阻力 f = -k*m*v*|v|，经 qfrc_applied 施加（世界系三个平动 dof）。

    注意：freejoint 平动 qvel 就是世界系线速度（MuJoCo freejoint 约定）。
    每个物理子步调用一次（在 mj_step 之前）。
    """
    if k <= 0:
        return
    v = mj_data.qvel[ball_dof:ball_dof + 3]
    speed = np.linalg.norm(v)
    if speed < 1e-9:
        return
    m = mj_model.body_mass[ball_body_id]
    mj_data.qfrc_applied[ball_dof:ball_dof + 3] = -k * m * v * speed


def step_with_drag(sim, state, motor, ball_dof: int, ball_body_id: int,
                   k: float):
    """带二次阻力的控制步（与 MJSim.step 相同的 PD + 子步循环，
    每个物理子步前对球施加 drag 外力）。"""
    for _ in range(sim.num_sim_substeps):
        apply_drag(state.mj_data, ball_dof, ball_body_id, sim.mj_model, k)
        mj_data = state.mj_data
        if sim.mj_model.nq > 36:
            qpos_jnt = mj_data.qpos[7:36]
            qvel_jnt = mj_data.qvel[6:35]
        else:
            qpos_jnt = mj_data.qpos[7:]
            qvel_jnt = mj_data.qvel[6:]
        torques = sim.kps * (np.asarray(motor).flatten() - qpos_jnt) \
            + sim.kds * (-qvel_jnt)
        mj_data.ctrl[:] = np.clip(torques, -sim.torque_limit, sim.torque_limit)
        mujoco.mj_step(sim.mj_model, mj_data)
    state.mj_data.qfrc_applied[:] = 0.0
    return state
