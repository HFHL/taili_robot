"""环境配置模块 —— 所有超参数集中在此，逻辑代码中禁止硬编码。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from urdf.taili_quad import (
    ACTUATED_JOINT_NAMES,
    DEFAULT_BASE_POS,
    DEFAULT_BASE_QUAT,
    DEFAULT_JOINT_ANGLES,
    URDF_PATH,
)


def _default_joint_names() -> List[str]:
    return list(ACTUATED_JOINT_NAMES)


def _default_joint_angles() -> Dict[str, float]:
    return dict(DEFAULT_JOINT_ANGLES)


@dataclass
class SimCfg:
    """物理仿真相关配置。"""

    num_envs: int = 4096
    sim_dt: float = 0.02
    substeps: int = 2
    episode_length_s: float = 20.0
    env_spacing: Tuple[float, float] = (2.0, 2.0)
    # 训练可复现；传入 gs.init(seed=...) 与 envs/seeding.set_global_seed
    seed: int = 42


@dataclass
class ActionCfg:
    """
    动作空间与 PD 位置控制映射。

    策略输出 a ∈ [-clip_actions, clip_actions]^N，映射为
    q_target = q_default + a * action_scale（逐关节可配）。
    """

    # 策略输出裁剪（rsl_rl 默认高斯策略配合 clip_actions=1.0）
    clip_actions: float = 1.0
    # 统一缩放 [rad]；hip/thigh/calf 差异大时可设 action_scale_per_joint
    action_scale: float = 0.25
    action_scale_per_joint: Dict[str, float] | None = None
    # 目标角相对 URDF 限位的收缩比例（0.9 表示使用限位区间中间 90%）
    soft_joint_pos_limit_factor: float = 0.9
    # 是否执行上一帧动作（1 步延迟，更贴近真实机）
    simulate_action_latency: bool = True
    # 当前仅支持位置 PD
    control_mode: str = "position"


@dataclass
class RobotCfg:
    """机器人 URDF 与 PD 增益配置。"""

    urdf_path: str = str(URDF_PATH)
    joint_names: List[str] = field(default_factory=_default_joint_names)
    default_joint_angles: Dict[str, float] = field(default_factory=_default_joint_angles)
    base_init_pos: Tuple[float, float, float] = DEFAULT_BASE_POS
    base_init_quat: Tuple[float, float, float, float] = DEFAULT_BASE_QUAT
    kp: float = 80.0
    kd: float = 2.0


@dataclass
class ObsCfg:
    """观测空间配置。"""

    include_lin_vel: bool = True
    include_ang_vel: bool = True
    include_projected_gravity: bool = True
    include_dof_pos: bool = True
    include_dof_vel: bool = True
    include_actions: bool = True
    # 策略输入是否包含速度指令（locomotion 建议 True）
    include_commands: bool = True

    obs_scales: Dict[str, float] = field(default_factory=lambda: {
        "lin_vel": 2.0,
        "ang_vel": 0.25,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
    })

    def num_obs(self, num_actions: int, num_commands: int = 3) -> int:
        from envs.observations import observation_dim

        return observation_dim(self, num_actions, num_commands)


@dataclass
class CommandCfg:
    """速度跟踪指令 [vx, vy, yaw_rate]（机体系 / 世界系前向，与 Go2 一致）。"""

    num_commands: int = 3
    lin_vel_x_range: Tuple[float, float] = (0.5, 0.5)
    lin_vel_y_range: Tuple[float, float] = (0.0, 0.0)
    ang_vel_range: Tuple[float, float] = (0.0, 0.0)
    resampling_time_s: float = 4.0


@dataclass
class RewardCfg:
    """奖励函数权重与辅助参数（权重会乘 sim_dt）。"""

    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "alive": 1.0,
        "tracking_lin_vel": 1.0,
        "tracking_ang_vel": 0.2,
    })
    tracking_sigma: float = 0.25


@dataclass
class TerminationCfg:
    """终止条件配置。"""

    max_roll_deg: float = 45.0
    max_pitch_deg: float = 45.0
    min_base_height: float = 0.15
    max_xy_distance: float = 15.0
    terminate_on_base_contact: bool = True
    base_contact_link_name: str = "base_link"


@dataclass
class EnvCfg:
    """顶层环境配置，聚合上述所有子配置。"""

    sim: SimCfg = field(default_factory=SimCfg)
    robot: RobotCfg = field(default_factory=RobotCfg)
    action: ActionCfg = field(default_factory=ActionCfg)
    obs: ObsCfg = field(default_factory=ObsCfg)
    command: CommandCfg = field(default_factory=CommandCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    termination: TerminationCfg = field(default_factory=TerminationCfg)
    show_viewer: bool = False
    terrain_urdf_path: str = "urdf/plane/plane.urdf"

    @property
    def num_actions(self) -> int:
        return len(self.robot.joint_names)

    @property
    def num_obs(self) -> int:
        return self.obs.num_obs(self.num_actions, self.command.num_commands)

    @property
    def max_episode_length(self) -> int:
        import math

        return math.ceil(self.sim.episode_length_s / self.sim.sim_dt)
