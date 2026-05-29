"""Version 1 观测 —— Actor 48 维 + Critic 特权 7 维。"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from train.version1.cfg import V1EnvCfg
    from train.version1.env import GenesisEnvV1


def policy_obs_dim(cfg: V1EnvCfg) -> int:
    n = cfg.num_actions
    dim = 0
    if cfg.obs.include_ang_vel:
        dim += 3
    if cfg.obs.include_projected_gravity:
        dim += 3
    if cfg.obs.include_dof_pos:
        dim += n
    if cfg.obs.include_dof_vel:
        dim += n
    if cfg.obs.include_actions:
        dim += n
    if cfg.obs.include_commands:
        dim += cfg.command.num_velocity_commands
    if cfg.obs.include_mob_commands:
        dim += cfg.command.num_mob_commands
    return dim


def policy_obs_component_names(cfg: V1EnvCfg) -> List[str]:
    n = cfg.num_actions
    names: List[str] = []
    if cfg.obs.include_ang_vel:
        names.append("base_ang_vel(3)")
    if cfg.obs.include_projected_gravity:
        names.append("projected_gravity(3)")
    if cfg.obs.include_dof_pos:
        names.append(f"dof_pos_error({n})")
    if cfg.obs.include_dof_vel:
        names.append(f"dof_vel({n})")
    if cfg.obs.include_actions:
        names.append(f"history_actions({n})")
    if cfg.obs.include_commands:
        names.append(f"commands({cfg.command.num_velocity_commands})")
    if cfg.obs.include_mob_commands:
        names.append(f"mob_commands({cfg.command.num_mob_commands})")
    return names


def critic_obs_component_names(cfg: V1EnvCfg) -> List[str]:
    return policy_obs_component_names(cfg) + [
        "privileged_base_lin_vel(3)",
        "privileged_friction(1)",
        "privileged_com_offset(3)",
    ]


def write_policy_observations(env: GenesisEnvV1, obs_buf: torch.Tensor) -> None:
    cfg = env.cfg.obs
    scales = cfg.obs_scales
    off = 0

    if cfg.include_ang_vel:
        obs_buf[:, off : off + 3] = env.base_ang_vel * scales["ang_vel"]
        off += 3
    if cfg.include_projected_gravity:
        obs_buf[:, off : off + 3] = env.projected_gravity
        off += 3
    if cfg.include_dof_pos:
        obs_buf[:, off : off + env.num_actions] = (env.dof_pos - env.default_dof_pos) * scales["dof_pos"]
        off += env.num_actions
    if cfg.include_dof_vel:
        obs_buf[:, off : off + env.num_actions] = env.dof_vel * scales["dof_vel"]
        off += env.num_actions
    if cfg.include_actions:
        obs_buf[:, off : off + env.num_actions] = env.actions
        off += env.num_actions
    if cfg.include_commands:
        nc = env.cfg.command.num_velocity_commands
        obs_buf[:, off : off + nc] = env.velocity_commands * scales["commands"]
        off += nc
    if cfg.include_mob_commands:
        nm = env.cfg.command.num_mob_commands
        mob_scaled = torch.stack(
            (
                env.mob_commands[:, 0] * scales["mob_freq"],
                env.mob_commands[:, 1] * scales["mob_height"],
                env.mob_commands[:, 2] * scales["mob_phase"],
            ),
            dim=-1,
        )
        obs_buf[:, off : off + nm] = mob_scaled
        off += nm

    assert off == obs_buf.shape[1]


def write_critic_observations(env: GenesisEnvV1, critic_buf: torch.Tensor) -> None:
    write_policy_observations(env, critic_buf[:, : env.cfg.num_obs])
    off = env.cfg.num_obs
    scales = env.cfg.obs.obs_scales
    critic_buf[:, off : off + 3] = env.base_lin_vel * scales["lin_vel"]
    off += 3
    critic_buf[:, off : off + 1] = env.privileged_friction
    off += 1
    critic_buf[:, off : off + 3] = env.privileged_com_offset
    off += 3
    assert off == critic_buf.shape[1]


def clip_observations(buf: torch.Tensor, clip_val: float) -> None:
    if clip_val > 0:
        buf.clamp_(-clip_val, clip_val)
