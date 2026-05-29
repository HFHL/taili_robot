"""taili_quad 机器人 URDF 资源包。"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
URDF_PATH = PACKAGE_ROOT / "urdf" / "taili_quad.urdf"
MESHES_DIR = PACKAGE_ROOT / "meshes"

# 12 个受控转动关节（与 RL 配置、Isaac Lab 生成物一致）
ACTUATED_JOINT_NAMES = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)

FOOT_LINK_NAMES = ("FR_foot_Link", "FL_foot_Link", "RR_foot_Link", "RL_foot_Link")

# 默认站立姿态（与 Isaac Lab / load_viewer 一致）
DEFAULT_BASE_POS = (0.0, 0.0, 0.58)
DEFAULT_BASE_QUAT = (1.0, 0.0, 0.0, 0.0)
DEFAULT_JOINT_ANGLES: dict[str, float] = {
    "FR_hip_joint": 0.0,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -0.5,
    "FL_hip_joint": 0.0,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -0.5,
    "RR_hip_joint": 0.0,
    "RR_thigh_joint": 0.8,
    "RR_calf_joint": -0.5,
    "RL_hip_joint": 0.0,
    "RL_thigh_joint": 0.8,
    "RL_calf_joint": -0.5,
}

# 对称性检查：同类型连杆质量配对（左/右、前/后）
SYMMETRY_MASS_PAIRS = (
    ("FR_hip_Link", "FL_hip_Link"),
    ("RR_hip_Link", "RL_hip_Link"),
    ("FR_thigh_Link", "FL_thigh_Link"),
    ("RR_thigh_Link", "RL_thigh_Link"),
    ("FR_calf_Link", "FL_calf_Link"),
    ("RR_calf_Link", "RL_calf_Link"),
    ("FR_foot_Link", "FL_foot_Link"),
    ("RR_foot_Link", "RL_foot_Link"),
)
