"""Version 1 指令采样 —— 20% 站立 + 速度/MoB 域采样。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import genesis as gs

if TYPE_CHECKING:
    from train.version1.cfg import V1CommandCfg
    from train.version1.env import GenesisEnvV1


def _uniform(n: int, low: float, high: float, device: torch.device, generator: torch.Generator | None) -> torch.Tensor:
    return (high - low) * torch.rand((n,), dtype=gs.tc_float, device=device, generator=generator) + low


def sample_velocity_commands(
    cfg: V1CommandCfg,
    n: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """采样 [n, 3]：vx, vy, yaw_rate。"""
    vx = _uniform(n, *cfg.lin_vel_x_range, device, generator)
    vy = _uniform(n, *cfg.lin_vel_y_range, device, generator)
    wz = _uniform(n, *cfg.ang_vel_range, device, generator)
    return torch.stack((vx, vy, wz), dim=-1)


def sample_mob_commands(
    cfg: V1CommandCfg,
    n: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """采样 [n, 3]：步频、目标机身高度、相位。"""
    freq = _uniform(n, *cfg.freq_range, device, generator)
    height = _uniform(n, *cfg.height_range, device, generator)
    phase = _uniform(n, *cfg.phase_range, device, generator)
    return torch.stack((freq, height, phase), dim=-1)


def resample_commands_v1(env: GenesisEnvV1, envs_idx: torch.Tensor | None) -> None:
    """Reset / 超时后重采样；20% 环境为绝对静止站立。"""
    cfg = env.cfg.command
    gen = getattr(env, "command_generator", None)

    if envs_idx is None:
        n = env.num_envs
        env_ids = slice(None)
    else:
        n = envs_idx.numel()
        env_ids = envs_idx

    vel = sample_velocity_commands(cfg, n, env.device, generator=gen)
    mob = sample_mob_commands(cfg, n, env.device, generator=gen)

    stand_mask = torch.rand(n, device=env.device, generator=gen) < cfg.stand_prob
    vel[stand_mask, :] = 0.0
    mob[stand_mask, 0] = 0.0

    env.velocity_commands[env_ids] = vel
    env.mob_commands[env_ids] = mob
