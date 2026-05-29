"""速度指令采样 —— 供跟踪奖励与（可选）观测使用。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

import genesis as gs

if TYPE_CHECKING:
    from configs.env_cfg import CommandCfg
    from envs.genesis_env import GenesisEnv


def command_limits(cfg: CommandCfg, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回 (lower, upper)，形状均为 [3]：vx, vy, yaw_rate。"""
    lower = torch.tensor(
        [cfg.lin_vel_x_range[0], cfg.lin_vel_y_range[0], cfg.ang_vel_range[0]],
        dtype=gs.tc_float,
        device=device,
    )
    upper = torch.tensor(
        [cfg.lin_vel_x_range[1], cfg.lin_vel_y_range[1], cfg.ang_vel_range[1]],
        dtype=gs.tc_float,
        device=device,
    )
    return lower, upper


def sample_commands(
    cfg: CommandCfg,
    num_envs: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """在配置范围内均匀采样 [num_envs, 3]，张量位于 device（MPS/CUDA/CPU）。"""
    lower, upper = command_limits(cfg, device)
    return (
        (upper - lower) * torch.rand((num_envs, 3), dtype=gs.tc_float, device=device, generator=generator)
        + lower
    )


def resample_commands(env: GenesisEnv, envs_idx: torch.Tensor | None) -> None:
    """为指定环境重采样速度指令（仅在 gs.device 上写张量）。"""
    cfg = env.cfg.command
    gen = getattr(env, "command_generator", None)
    if envs_idx is None:
        env.commands.copy_(sample_commands(cfg, env.num_envs, env.device, generator=gen))
    else:
        new_cmds = sample_commands(cfg, envs_idx.numel(), env.device, generator=gen)
        env.commands[envs_idx] = new_cmds
