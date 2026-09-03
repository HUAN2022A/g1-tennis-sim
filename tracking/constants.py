import os
from pathlib import Path
import numpy as np
from utils.path import PATH_ASSET
# HAND 选择安装的末端执行器（None / Default / Dex3 / BrainCo），
# 由手部感知的模型加载器（utils.hand.load_mj_model）使用。
# 在此处重新导出，以便训练堆栈的其余部分可以读取 consts.HAND。
from utils.hand import HAND, HAND_INFO, load_mj_model, load_mj_spec  # noqa: F401

# G1 机器人硬件版本，通过环境变量 G1_VERSION 设置（默认 5010）
# 不同版本的 G1 可能有不同的关节配置或物理参数
G1_VERSION = os.environ.get("G1_VERSION", "5010")
ROOT_PATH = PATH_ASSET / f"unitree_g1_{G1_VERSION}"

# MuJoCo XML 模型文件路径
FEET_ONLY_FLAT_TERRAIN_XML = ROOT_PATH / "scene_mjx_feetonly_flat_terrain.xml"  # 仅脚部碰撞（平地）
FEET_ONLY_ROUGH_TERRAIN_XML = ROOT_PATH / "scene_mjx_feetonly_rough_terrain.xml"  # 仅脚部碰撞（崎岖地形）
FULL_COLLISIONS_XML = ROOT_PATH / "scene_mjx.xml"  # 全身碰撞
BODY_POSE_XML = ROOT_PATH / "scene_mjx_body_pose.xml"  # 身体姿态估计
LOCO_XML = ROOT_PATH / "scene_mjx_loco.xml"  # 运动（locomotion）
TRACK_XML = ROOT_PATH / "scene_mjx_track.xml"  # 运动跟踪（tracking）
DEBUG_TRACK_XML = ROOT_PATH / "scene_mjx_track_debug.xml"  # 调试用的跟踪场景


def task_to_xml(task_name: str) -> Path:
    """根据任务名称返回对应的 MuJoCo XML 文件路径"""
    return {
        "flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,
        "rough_terrain": FEET_ONLY_ROUGH_TERRAIN_XML,
        "full_collision": FULL_COLLISIONS_XML,
        "body_pose": BODY_POSE_XML,
        "locomotion_flat": LOCO_XML,
        "track": TRACK_XML,
    }[task_name]


# 脚部站点（site）名称，用于接触检测
FEET_SITES = [
    "left_foot",
    "right_foot",
]

# 手部站点名称（用于灵巧手跟踪）
HAND_SITES = [
    "left_palm",
    "right_palm",
]

# 左右脚的几何体名称，用于碰撞检测
LEFT_FEET_GEOMS = [
    "left_foot1_collision",
    "left_foot2_collision",
    "left_foot3_collision",
]
RIGHT_FEET_GEOMS = [
    "right_foot1_collision",
    "right_foot2_collision",
    "right_foot3_collision",
]

ROOT_BODY = "torso_link"  # 根节点 body 名称

# 传感器名称定义
GRAVITY_SENSOR = "upvector"  # 重力向量传感器
GLOBAL_LINVEL_SENSOR = "global_linvel"  # 全局线速度传感器
GLOBAL_ANGVEL_SENSOR = "global_angvel"  # 全局角速度传感器
LOCAL_LINVEL_SENSOR = "local_linvel"  # 局部线速度传感器
ACCELEROMETER_SENSOR = "accelerometer"  # 加速度计
GYRO_SENSOR = "gyro"  # 陀螺仪

#############
#   关节定义   #
#############

NUM_JOINT = 29  # G1 机器人的关节数量（全身自由度）


class MotorID:
    """电机 ID 映射：将关节名称映射到数组索引"""
    LEG_L = [0, 1, 2, 3, 4, 5]  # 左腿：6 个关节
    LEG_R = [6, 7, 8, 9, 10, 11]  # 右腿：6 个关节
    WAIST = [12, 13, 14]  # 腰部：3 个关节
    ARM_L = [15, 16, 17, 18, 19, 20, 21]  # 左臂：7 个关节
    ARM_R = [22, 23, 24, 25, 26, 27, 28]  # 右臂：7 个关节
    LEGs = LEG_L + LEG_R  # 所有腿部关节
    ARMs = ARM_L + ARM_R  # 所有手臂关节
    FULL = LEG_L + LEG_R + WAIST + ARM_L + ARM_R  # 全部 29 个关节


class MotorName:
    """电机名称映射：G1 机器人各关节的完整名称

    关节命名规则：<左右>_<部位>_<类型>_joint
    例如：left_hip_pitch_joint 表示左腿髋关节的俯仰关节
    """
    LEG_L = [
        "left_hip_pitch_joint",   # 左髋俯仰
        "left_hip_roll_joint",    # 左髋横滚
        "left_hip_yaw_joint",     # 左髋偏航
        "left_knee_joint",        # 左膝
        "left_ankle_pitch_joint", # 左踝俯仰
        "left_ankle_roll_joint",  # 左踝横滚
    ]
    LEG_R = [
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ]
    WAIST = [
        "waist_yaw_joint",   # 腰部偏航
        "waist_roll_joint",  # 腰部横滚
        "waist_pitch_joint", # 腰部俯仰
    ]
    ARM_L = [
        "left_shoulder_pitch_joint",  # 左肩俯仰
        "left_shoulder_roll_joint",   # 左肩横滚
        "left_shoulder_yaw_joint",    # 左肩偏航
        "left_elbow_joint",           # 左肘
        "left_wrist_roll_joint",      # 左腕横滚
        "left_wrist_pitch_joint",     # 左腕俯仰
        "left_wrist_yaw_joint",       # 左腕偏航
    ]
    ARM_R = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    FULL = LEG_L + LEG_R + WAIST + ARM_L + ARM_R  # 全部 29 个关节


# 受限关节范围（软限位），用于防止关节超出安全范围
# 格式：(最小值, 最大值)，单位：弧度
RESTRICTED_JOINT_RANGE = (
    # 左腿
    (-1.57, 1.57),   # 髋俯仰
    (-0.5, 0.5),     # 髋横滚
    (-0.7, 0.7),     # 髋偏航
    (0, 1.57),       # 膝（只能弯曲，不能反弯）
    (-0.4, 0.4),     # 踝俯仰
    (-0.2, 0.2),     # 踝横滚
    # 右腿
    (-1.57, 1.57),
    (-0.5, 0.5),
    (-0.7, 0.7),
    (0, 1.57),
    (-0.4, 0.4),
    (-0.2, 0.2),
    # 腰部
    (-2.618, 2.618),  # 腰偏航（约 ±150 度）
    (-0.52, 0.52),    # 腰横滚
    (-0.52, 0.52),    # 腰俯仰
    # 左肩
    (-1.57, 0.520),
    (0.000, 1.570),
    (-1.57, 0.000),
    (-0.52, 1.570),
    (-1.57, 1.570),
    (-0.52, 0.520),
    (-0.52, 0.520),
    # 右肩
    (-1.57, 0.520),
    (-1.57, 0.000),
    (0.000, 1.570),
    (-0.52, 1.570),
    (-1.57, 1.570),
    (-0.52, 0.520),
    (-0.52, 0.520),
)


DEFAULT_CHEST_Z = 1.05  # 默认胸部高度（米）

# 各关节的力矩限制（单位：Nm）
# 顺序与 MotorName.FULL 一致
TORQUE_LIMIT = np.array([
    88., 139., 88., 139., 50., 50.,      # 左腿
    88., 139., 88., 139., 50., 50.,      # 右腿
    88., 50., 50.,                        # 腰部
    25., 25., 25., 25., 25., 5., 5.,      # 左臂
    25., 25., 25., 25., 25., 5., 5.,      # 右臂
])

# 默认关节位置（站姿）
# 形状：(36,) = 7(基座：xyz + quat) + 29(关节)
DEFAULT_QPOS = np.float32([
	0, 0, 0.78,      # 基座位置 (x, y, z)，z=0.78m 是站立高度
	1, 0, 0, 0,      # 基座四元数 (w, x, y, z)，单位四元数表示无旋转
	-0.1, 0, 0, 0.3, -0.2, 0,    # 左腿：髋俯仰微屈、膝盖弯曲、踝俯仰补偿
	-0.1, 0, 0, 0.3, -0.2, 0,    # 右腿：与左腿对称
	0, 0, 0,                      # 腰部：保持直立（yaw, roll, pitch）
	0.2, 0.3, 0, 1.28, 0, 0, 0,   # 左臂：肩部微张、肘部弯曲约 73 度
	0.2,-0.3, 0, 1.28, 0, 0, 0,   # 右臂：与左臂对称
])


# 基础 PD 增益（比例增益），用于位置控制
# 数值经过调优，平衡跟踪精度和稳定性
BASE_KPs = np.float32([
    40.17923737, 99.09842682, 40.17923737, 99.09842682, 28.5012455 ,
    28.5012455 , 40.17923737, 99.09842682, 40.17923737, 99.09842682,
    28.5012455 , 28.5012455 , 40.17923737, 28.5012455 , 28.5012455 ,
    14.25062275, 14.25062275, 14.25062275, 14.25062275, 14.25062275,
    16.77832794, 16.77832794, 14.25062275, 14.25062275, 14.25062275,
    14.25062275, 14.25062275, 16.77832794, 16.77832794
])

BASE_KDs = np.float32([
    2.5578897 , 6.30880165, 2.5578897 , 6.30880165, 1.81444573,
    1.81444573, 2.5578897 , 6.30880165, 2.5578897 , 6.30880165,
    1.81444573, 1.81444573, 2.5578897 , 1.81444573, 1.81444573,
    0.90722287, 0.90722287, 0.90722287, 0.90722287, 0.90722287,
    1.06814146, 1.06814146, 0.90722287, 0.90722287, 0.90722287,
    0.90722287, 0.90722287, 1.06814146, 1.06814146
])

# 上半身关节列表（手臂部分）
UPPER_BODY_JOINTs = [
    # 左臂
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # 右臂
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# 关键点（Keypoint）定义：下半身
KPT_LOWER = [
    "pelvis",                # 骨盆
    "left_hip_roll_link",    # 左髋横滚连杆
    "left_knee_link",        # 左膝连杆
    "left_ankle_roll_link",  # 左踝横滚连杆
    "right_hip_roll_link",   # 右髋横滚连杆
    "right_knee_link",       # 右膝连杆
    "right_ankle_roll_link", # 右踝横滚连杆
]
# 关键点（Keypoint）定义：上半身
KPT_UPPER = [
    "torso_link",                # 躯干
    "left_shoulder_roll_link",   # 左肩横滚连杆
    "left_elbow_link",           # 左肘连杆
    "left_wrist_yaw_link",       # 左腕偏航连杆
    "right_shoulder_roll_link",  # 右肩横滚连杆
    "right_elbow_link",          # 右肘连杆
    "right_wrist_yaw_link",      # 右腕偏航连杆
]
# 所有关键点名称（共 14 个：下半身 7 + 上半身 7）
# 用于运动跟踪的关键部位，策略网络学习跟踪这些关键点的位姿
KPT_NAMES = [
    # 下半身
    "pelvis",                # 骨盆（根节点）
    "left_hip_roll_link",    # 左髋
    "left_knee_link",        # 左膝
    "left_ankle_roll_link",  # 左踝
    "right_hip_roll_link",   # 右髋
    "right_knee_link",       # 右膝
    "right_ankle_roll_link", # 右踝
    # 上半身
    "torso_link",                # 躯干
    "left_shoulder_roll_link",   # 左肩
    "left_elbow_link",           # 左肘
    "left_wrist_yaw_link",       # 左腕
    "right_shoulder_roll_link",  # 右肩
    "right_elbow_link",          # 右肘
    "right_wrist_yaw_link",      # 右腕
]
# 末端关键点（脚踝和手腕），用于计算跟踪误差
KPT_END_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

# 脚部连杆名称，用于接触检测
FEET_LINKs = ["left_ankle_roll_link", "right_ankle_roll_link"]

# 各关节的速度限制（单位：rad/s）
DOF_VEL_LIMITS = [
    32.0, 32.0, 32.0, 20.0, 37.0, 37.0,  # 左腿
    32.0, 32.0, 32.0, 20.0, 37.0, 37.0,  # 右腿
    32.0, 37.0, 37.0,                     # 腰部
    37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0,  # 左臂
    37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0   # 右臂
]

# 动作关节名称：策略网络控制的 29 个关节
# 这些关节接收神经网络输出的目标位置，通过 PD 控制器驱动
ACTION_JOINT_NAMES = [
    # 左腿
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # 右腿
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # -------------- 仅用于跟踪 --------------
    # 腰部
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # 左臂
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # 右臂
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
# 观测关节名称：通常与动作关节相同
OBS_JOINT_NAMES = ACTION_JOINT_NAMES

# 以下是历史版本的关节配置（用于兼容不同的模型检查点）
# 66177: 排除腰部 roll/pitch 和部分腕关节
ACTION_JOINT_NAMES_66177 = [
    # left leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # -------------- tracking only --------------
    # waist
    "waist_yaw_joint",
    # "waist_roll_joint",  # 13
    # "waist_pitch_joint",  # 14
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",  # 20
    "left_wrist_yaw_joint",  # 21
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",  # 27
    "right_wrist_yaw_joint",  # 28
]


ACTION_JOINT_NAMES_66155 = [
    # left leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # -------------- tracking only --------------
    # waist
    "waist_yaw_joint",
    # "waist_roll_joint",  # 13
    # "waist_pitch_joint",  # 14
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    # "left_wrist_pitch_joint",  # 20
    # "left_wrist_yaw_joint",  # 21
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    # "right_wrist_pitch_joint",  # 27
    # "right_wrist_yaw_joint",  # 28
]


ACTION_JOINT_NAMES_66144 = [
    # left leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # -------------- tracking only --------------
    # waist
    "waist_yaw_joint",
    # "waist_roll_joint",  # 13
    # "waist_pitch_joint",  # 14
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    # "left_wrist_roll_joint",
    # "left_wrist_pitch_joint",  # 20
    # "left_wrist_yaw_joint",  # 21
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    # "right_wrist_roll_joint",
    # "right_wrist_pitch_joint",  # 27
    # "right_wrist_yaw_joint",  # 28
]

BASE_KP_KD_SCALE = float(os.environ.get("BASE_KP_KD_SCALE", "1.0"))

JOINT_KP_KD_SCALE = {
    # left leg
    "left_hip_pitch_joint": 1,
    "left_hip_roll_joint": 1,
    "left_hip_yaw_joint": 1,
    "left_knee_joint": 1,
    "left_ankle_pitch_joint": 1,
    "left_ankle_roll_joint": 1,
    # right leg
    "right_hip_pitch_joint": 1,
    "right_hip_roll_joint": 1,
    "right_hip_yaw_joint": 1,
    "right_knee_joint": 1,
    "right_ankle_pitch_joint": 1,
    "right_ankle_roll_joint": 1,
    # waist
    "waist_yaw_joint": 1,
    "waist_roll_joint": 1,
    "waist_pitch_joint": 1,
    # left arm
    "left_shoulder_pitch_joint": 1,
    "left_shoulder_roll_joint": 1,
    "left_shoulder_yaw_joint": 1,
    "left_elbow_joint": 1,
    "left_wrist_roll_joint": 1,
    "left_wrist_pitch_joint": 1,
    "left_wrist_yaw_joint": 1,
    # 右臂
    "right_shoulder_pitch_joint": 1,
    "right_shoulder_roll_joint": 1,
    "right_shoulder_yaw_joint": 1,
    "right_elbow_joint": 1,
    "right_wrist_roll_joint": 1,
    "right_wrist_pitch_joint": 1,
    "right_wrist_yaw_joint": 1,
}

# 将字典转换为 NumPy 数组，顺序与 ACTION_JOINT_NAMES 一致
JOINT_KP_KD_SCALE = np.array(list(JOINT_KP_KD_SCALE.values()), dtype=np.float32)

# 最终的 PD 增益计算
# KPs: 比例增益，考虑基础增益、关节特定缩放和全局缩放
# KDs: 微分增益，同样考虑所有缩放因子
KPs = BASE_KPs * JOINT_KP_KD_SCALE * BASE_KP_KD_SCALE ** 2
KDs = BASE_KDs * JOINT_KP_KD_SCALE * BASE_KP_KD_SCALE

# 动作缩放因子：根据力矩限制和 PD 增益计算
# 用于将神经网络输出（通常在 [-1, 1] 范围）转换为实际的关节位置偏移
ACTION_SCALE = TORQUE_LIMIT / KPs
