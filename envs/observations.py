"""
观测空间定义 —— 将仿真状态拼接为 [num_envs, num_obs] 策略输入。

默认拼接顺序（与 Go2 locomotion 对齐，不含速度指令）::

    [0:3]   base_lin_vel        机体系线速度 × obs_scales["lin_vel"]
    [3:6]   base_ang_vel        机体系角速度 × obs_scales["ang_vel"]
    [6:9]   projected_gravity   机体系重力方向单位向量（无额外缩放）
    [9:9+N] dof_pos             (q - q_default) × obs_scales["dof_pos"]
    [9+N:9+2N] dof_vel          qdot × obs_scales["dof_vel"]
    [9+2N:9+3N] actions         上一步策略输出（未缩放）

其中 N = num_actions（taili_quad 为 12）。含速度指令时 +3，
默认 num_obs = 9 + 3×12 + 3 = 48。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from configs.env_cfg import ObsCfg
    from envs.genesis_env import GenesisEnv


def observation_dim(cfg: ObsCfg, num_actions: int, num_commands: int = 3) -> int:
    """根据 ObsCfg 开关计算观测维度。"""
    n = 0
    if cfg.include_lin_vel:
        n += 3
    if cfg.include_ang_vel:
        n += 3
    if cfg.include_projected_gravity:
        n += 3
    if cfg.include_dof_pos:
        n += num_actions
    if cfg.include_dof_vel:
        n += num_actions
    if cfg.include_actions:
        n += num_actions
    if cfg.include_commands:
        n += num_commands
    return n


def observation_component_names(
    cfg: ObsCfg, num_actions: int, num_commands: int = 3
) -> List[str]:
    """返回各段观测的名称与长度，便于日志 / 调试。"""
    names: List[str] = []
    if cfg.include_lin_vel:
        names.append("base_lin_vel(3)")
    if cfg.include_ang_vel:
        names.append("base_ang_vel(3)")
    if cfg.include_projected_gravity:
        names.append("projected_gravity(3)")
    if cfg.include_dof_pos:
        names.append(f"dof_pos({num_actions})")
    if cfg.include_dof_vel:
        names.append(f"dof_vel({num_actions})")
    if cfg.include_actions:
        names.append(f"actions({num_actions})")
    if cfg.include_commands:
        names.append(f"commands({num_commands})")
    return names


def write_observations(env: GenesisEnv, obs_buf: torch.Tensor) -> None:
    """
    将 env 内已更新的状态 buffer 写入 obs_buf（原地赋值）。

    要求 env 已维护: base_lin_vel, base_ang_vel, projected_gravity,
    dof_pos, default_dof_pos, dof_vel, actions。
    """
    cfg = env.cfg.obs
    scales = cfg.obs_scales
    off = 0

    if cfg.include_lin_vel:
        obs_buf[:, off : off + 3] = env.base_lin_vel * scales["lin_vel"]
        off += 3
    if cfg.include_ang_vel:
        obs_buf[:, off : off + 3] = env.base_ang_vel * scales["ang_vel"]
        off += 3
    if cfg.include_projected_gravity:
        obs_buf[:, off : off + 3] = env.projected_gravity
        off += 3
    if cfg.include_dof_pos:
        obs_buf[:, off : off + env.num_actions] = (
            env.dof_pos - env.default_dof_pos
        ) * scales["dof_pos"]
        off += env.num_actions
    if cfg.include_dof_vel:
        obs_buf[:, off : off + env.num_actions] = env.dof_vel * scales["dof_vel"]
        off += env.num_actions
    if cfg.include_actions:
        obs_buf[:, off : off + env.num_actions] = env.actions
        off += env.num_actions
    if cfg.include_commands:
        n_cmd = env.cfg.command.num_commands
        obs_buf[:, off : off + n_cmd] = env.commands * env.commands_scale
        off += n_cmd

    assert off == obs_buf.shape[1], f"obs 写入长度 {off} 与 buffer 宽度 {obs_buf.shape[1]} 不一致"
