"""
张量化部分重置 —— 仅重置 done 的并行环境，其余环境状态不变。

与 ``GenesisEnv.step()`` 配合：
    reset_envs(env, reset_buf)   # reset_buf: bool [num_envs]

Genesis ``set_qpos`` 在部分 env 重置时需传入 bool 掩码 + 完整 batch qpos，
或 int 索引 + 对应子 batch；本模块采用与 go2_env 一致的 bool 掩码方式。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from envs.genesis_env import GenesisEnv


def as_env_mask(
    env_ids: torch.Tensor | None, num_envs: int, device: torch.device
) -> torch.Tensor | None:
    """
    将 env 选择器规整为 bool 掩码 [num_envs]，或 None 表示全部。

    接受 bool 掩码、int64 索引，或 None。
    """
    if env_ids is None:
        return None
    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, dtype=torch.int64, device=device)
    else:
        env_ids = env_ids.to(device=device)

    if env_ids.dtype == torch.bool:
        return env_ids.reshape(-1)
    if env_ids.numel() == 0:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)

    mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
    mask[env_ids.reshape(-1).to(dtype=torch.int64)] = True
    return mask


def env_mask_to_idx(env_mask: torch.Tensor) -> torch.Tensor:
    """bool 掩码 -> int64 索引（供 Genesis 部分 read API 使用）。"""
    return env_mask.nonzero(as_tuple=False).flatten()


def normalize_env_ids(env_ids: torch.Tensor | None, num_envs: int, device: torch.device) -> torch.Tensor | None:
    """将 env 选择器规整为 int64 索引（兼容旧调用）。"""
    env_mask = as_env_mask(env_ids, num_envs, device)
    if env_mask is None:
        return None
    idx = env_mask_to_idx(env_mask)
    return idx if idx.numel() > 0 else idx


def refresh_state_buffers(env: GenesisEnv, env_ids: torch.Tensor | None) -> None:
    """
    从 Genesis 读回状态，写入 env 的 GPU buffer。

    部分 reset 后必须调用，否则刚重置环境的 obs 仍使用旧 base_pos / dof_pos。
    """
    env_mask = as_env_mask(env_ids, env.num_envs, env.device)
    if env_mask is None:
        env_ids_int = None
    else:
        env_ids_int = env_mask_to_idx(env_mask)
        if env_ids_int.numel() == 0:
            return

    from genesis.utils.geom import inv_quat, transform_by_quat

    if env_ids_int is None:
        pos = env.robot.get_pos()
        quat = env.robot.get_quat()
        inv_q = inv_quat(quat)
        env.base_pos = pos
        env.base_quat = quat
        env.base_lin_vel = transform_by_quat(env.robot.get_vel(), inv_q)
        env.base_ang_vel = transform_by_quat(env.robot.get_ang(), inv_q)
        env.projected_gravity = transform_by_quat(env.global_gravity, inv_q)
        env.dof_pos = env.robot.get_dofs_position(env.motors_dof_idx)
        env.dof_vel = env.robot.get_dofs_velocity(env.motors_dof_idx)
        return

    pos = env.robot.get_pos(envs_idx=env_ids_int)
    quat = env.robot.get_quat(envs_idx=env_ids_int)
    inv_q = inv_quat(quat)

    env.base_pos[env_ids_int] = pos
    env.base_quat[env_ids_int] = quat
    env.base_lin_vel[env_ids_int] = transform_by_quat(env.robot.get_vel(envs_idx=env_ids_int), inv_q)
    env.base_ang_vel[env_ids_int] = transform_by_quat(env.robot.get_ang(envs_idx=env_ids_int), inv_q)
    env.projected_gravity[env_ids_int] = transform_by_quat(env.global_gravity, inv_q)
    env.dof_pos[env_ids_int] = env.robot.get_dofs_position(env.motors_dof_idx, envs_idx=env_ids_int)
    env.dof_vel[env_ids_int] = env.robot.get_dofs_velocity(env.motors_dof_idx, envs_idx=env_ids_int)


def reset_envs(env: GenesisEnv, env_ids: torch.Tensor | None) -> int:
    """
    重置指定并行环境（``env_ids is None`` 表示全部；bool / int 索引均可）。

    Returns
    -------
    int
        本次重置的环境数量。
    """
    env_mask = as_env_mask(env_ids, env.num_envs, env.device)
    n_reset = env.num_envs if env_mask is None else int(env_mask.sum().item())
    if n_reset == 0:
        return 0

    env.robot.set_qpos(
        env.init_qpos,
        envs_idx=env_mask,
        zero_velocity=True,
        skip_forward=True,
    )

    if env_mask is None:
        env.resample_commands(None)
    else:
        env.resample_commands(env_mask_to_idx(env_mask))

    _zero_episode_buffers(env, env_mask)
    refresh_state_buffers(env, env_mask)

    env._reset_episode_reward_sums(env_mask)

    if n_reset > 0:
        sums = getattr(env, "_episode_reward_sums", None)
        if sums:
            env.extras["episode"] = {
                f"rew_{name}": (buf[env_mask].mean() if env_mask is not None else buf.mean())
                for name, buf in sums.items()
            }
        env.extras["n_reset_envs"] = n_reset

    return n_reset


def _zero_episode_buffers(env: GenesisEnv, env_mask: torch.Tensor | None) -> None:
    """清零被重置环境的 episode 局部 buffer（不影响其它 env）。"""
    if env_mask is None:
        env.origin_xy.copy_(env.robot.get_pos()[:, :2])
        env.actions.zero_()
        env.last_actions.zero_()
        env.dof_vel.zero_()
        env.last_dof_vel.zero_()
        env.base_lin_vel.zero_()
        env.base_ang_vel.zero_()
        env.projected_gravity.copy_(env.init_projected_gravity)
        env.episode_length_buf.zero_()
        return

    env_ids_int = env_mask_to_idx(env_mask)
    env.origin_xy[env_mask] = env.robot.get_pos(envs_idx=env_ids_int)[:, :2]
    env.actions.masked_fill_(env_mask[:, None], 0.0)
    env.last_actions.masked_fill_(env_mask[:, None], 0.0)
    env.dof_vel.masked_fill_(env_mask[:, None], 0.0)
    env.last_dof_vel.masked_fill_(env_mask[:, None], 0.0)
    env.base_lin_vel.masked_fill_(env_mask[:, None], 0.0)
    env.base_ang_vel.masked_fill_(env_mask[:, None], 0.0)
    env.projected_gravity[env_mask] = env.init_projected_gravity
    env.episode_length_buf.masked_fill_(env_mask, 0)
