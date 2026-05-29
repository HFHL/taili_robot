"""Version 1 环境配置 —— 对应 train/version1/desing.md。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from configs.env_cfg import (
    ActionCfg,
    EnvCfg,
    ObsCfg,
    RewardCfg,
    RobotCfg,
    SimCfg,
    TerminationCfg,
)
from urdf.taili_quad import ACTUATED_JOINT_NAMES, DEFAULT_BASE_POS, DEFAULT_BASE_QUAT, DEFAULT_JOINT_ANGLES, URDF_PATH

VERSION_NAME = "version1"
VERSION_LABEL = "平地盲跑与站立 (Flat Ground Velocity Tracking)"


def _default_joint_names() -> list[str]:
    return list(ACTUATED_JOINT_NAMES)


def _default_joint_angles() -> Dict[str, float]:
    return dict(DEFAULT_JOINT_ANGLES)


@dataclass
class V1SimCfg(SimCfg):
    num_envs: int = 4096
    sim_dt: float = 0.005
    substeps: int = 1
    episode_length_s: float = 24.0
    decimation: int = 4
    seed: int = 42


@dataclass
class V1ActionCfg(ActionCfg):
    clip_actions: float = 100.0
    action_scale: float = 0.25
    soft_joint_pos_limit_factor: float = 0.9
    simulate_action_latency: bool = True


@dataclass
class V1RobotCfg(RobotCfg):
    urdf_path: str = str(URDF_PATH)
    joint_names: list[str] = field(default_factory=_default_joint_names)
    default_joint_angles: Dict[str, float] = field(default_factory=_default_joint_angles)
    base_init_pos: Tuple[float, float, float] = DEFAULT_BASE_POS
    base_init_quat: Tuple[float, float, float, float] = DEFAULT_BASE_QUAT
    kp: float = 20.0
    kd: float = 0.5
    torque_limit: float = 25.0


@dataclass
class V1ObsCfg(ObsCfg):
    """Actor 48 维：无特权线速度；Critic 另见 V1EnvCfg.num_privileged_obs。"""

    include_lin_vel: bool = False
    include_ang_vel: bool = True
    include_projected_gravity: bool = True
    include_dof_pos: bool = True
    include_dof_vel: bool = True
    include_actions: bool = True
    include_commands: bool = True
    include_mob_commands: bool = True
    clip_obs: float = 5.0

    obs_scales: Dict[str, float] = field(
        default_factory=lambda: {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "commands": 2.0,
            "mob_freq": 0.4,
            "mob_height": 3.0,
            "mob_phase": 0.318309886,
        }
    )


@dataclass
class V1CommandCfg:
    """速度指令 (3) + MoB 指令 (3)。"""

    num_commands: int = 3
    num_velocity_commands: int = 3
    num_mob_commands: int = 3
    stand_prob: float = 0.2
    lin_vel_x_range: Tuple[float, float] = (-1.0, 2.0)
    lin_vel_y_range: Tuple[float, float] = (-0.5, 0.5)
    ang_vel_range: Tuple[float, float] = (-1.0, 1.0)
    freq_range: Tuple[float, float] = (1.5, 3.5)
    height_range: Tuple[float, float] = (0.25, 0.35)
    phase_range: Tuple[float, float] = (0.0, 6.283185307)
    resampling_time_s: float = 1.0e9


@dataclass
class V1RewardCfg(RewardCfg):
    """乘法奖励 r_task * exp(0.02 * r_aux)；reward_weights 未使用。"""

    reward_weights: Dict[str, float] = field(default_factory=dict)
    aux_exp_scale: float = 0.02
    w_action_rate: float = -0.01
    w_torque: float = -0.0002
    w_phase: float = -1.0
    w_symmetry: float = -0.5
    w_dof_vel: float = -0.001
    task_sigma_vx: float = 0.1
    task_sigma_vy: float = 0.05
    task_sigma_wz: float = 0.05
    task_weight_vx: float = 0.5
    task_weight_vy: float = 0.3
    task_weight_wz: float = 0.2


@dataclass
class V1EnvCfg(EnvCfg):
    """Version 1 顶层配置。"""

    train_version: str = VERSION_NAME
    train_version_label: str = VERSION_LABEL
    sim: V1SimCfg = field(default_factory=V1SimCfg)
    robot: V1RobotCfg = field(default_factory=V1RobotCfg)
    action: V1ActionCfg = field(default_factory=V1ActionCfg)
    obs: V1ObsCfg = field(default_factory=V1ObsCfg)
    command: V1CommandCfg = field(default_factory=V1CommandCfg)
    reward: V1RewardCfg = field(default_factory=V1RewardCfg)
    termination: TerminationCfg = field(default_factory=TerminationCfg)
    symmetry_augmentation: bool = True
    terrain_static_friction: float = 1.0
    domain_rand_friction_range: Tuple[float, float] = (0.8, 1.2)
    domain_rand_com_offset_range: Tuple[float, float] = (-0.02, 0.02)

    @property
    def control_dt(self) -> float:
        return self.sim.sim_dt * self.sim.decimation

    @property
    def num_privileged_obs(self) -> int:
        return 7

    @property
    def num_obs(self) -> int:
        from train.version1.observations import policy_obs_dim

        return policy_obs_dim(self)

    @property
    def num_critic_obs(self) -> int:
        return self.num_obs + self.num_privileged_obs

    @property
    def max_episode_length(self) -> int:
        import math

        return math.ceil(self.sim.episode_length_s / self.control_dt)


def build_v1_env_cfg(
    num_envs: int = 4096,
    *,
    show_viewer: bool = False,
    seed: int = 42,
) -> V1EnvCfg:
    """按 desing.md 默认参数组装 V1 环境配置。"""
    return V1EnvCfg(
        sim=V1SimCfg(num_envs=num_envs, seed=seed),
        show_viewer=show_viewer,
    )
