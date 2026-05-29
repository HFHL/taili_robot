"""
终止条件 (Dones / Resets) —— 写入 env.reset_buf 与 extras['time_outs']。

默认检查项:
    - time_out: 达到 max_episode_length
    - fall_pose: 基座高度过低，或 roll/pitch 超过阈值（躯干倾倒）
    - base_contact: base_link 与地面存在接触（躯干触地）
    - far_from_origin: 水平方向偏离初始位置过远
    - sim_error: Genesis 刚体求解异常（穿模等）
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from genesis.utils.geom import quat_to_xyz, transform_quat_by_quat

if TYPE_CHECKING:
    from envs.genesis_env import GenesisEnv


def check_termination(env: GenesisEnv) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算本步终止掩码。

    Returns
    -------
    reset_buf : bool [num_envs]，True 表示 episode 结束并重置
    time_outs : float [num_envs]，1.0 表示因超时结束（供 rsl_rl bootstrap）
    """
    term = env.cfg.termination
    n = env.num_envs
    dev = env.device

    time_outs = (env.episode_length_buf >= env.max_episode_length).to(dtype=torch.float32, device=dev)
    reset_buf = time_outs.to(dtype=torch.bool)

    if term.min_base_height is not None:
        reset_buf |= env.base_pos[:, 2] < term.min_base_height

    if term.max_roll_deg is not None or term.max_pitch_deg is not None:
        base_euler_deg = _base_euler_deg(env)
        if term.max_roll_deg is not None:
            reset_buf |= torch.abs(base_euler_deg[:, 0]) > term.max_roll_deg
        if term.max_pitch_deg is not None:
            reset_buf |= torch.abs(base_euler_deg[:, 1]) > term.max_pitch_deg

    if term.max_xy_distance is not None:
        xy_dist = torch.linalg.norm(env.base_pos[:, :2] - env.origin_xy, dim=-1)
        reset_buf |= xy_dist > term.max_xy_distance

    if term.terminate_on_base_contact:
        reset_buf |= _base_link_contact(env)

    reset_buf |= env.scene.rigid_solver.get_error_envs_mask()

    return reset_buf, time_outs


def _base_euler_deg(env: GenesisEnv) -> torch.Tensor:
    """基座相对初始朝向的 roll, pitch, yaw [deg]，形状 [num_envs, 3]。"""
    rel_quat = transform_quat_by_quat(env.inv_base_init_quat, env.base_quat)
    return quat_to_xyz(rel_quat, rpy=True, degrees=True)


def _base_link_contact(env: GenesisEnv) -> torch.Tensor:
    """base_link 是否与任意碰撞体接触 [num_envs] bool。"""
    link_name = env.cfg.termination.base_contact_link_name
    base_link = env.robot.get_link(link_name)
    base_idx = base_link.idx

    contacts = env.robot.get_contacts()
    link_a = contacts["link_a"]
    link_b = contacts["link_b"]

    if link_a.dim() == 1:
        touched = (link_a == base_idx) | (link_b == base_idx)
        return touched.any().expand(env.num_envs)
    touched = (link_a == base_idx) | (link_b == base_idx)
    if "valid_mask" in contacts:
        touched = touched & contacts["valid_mask"]
    return touched.any(dim=-1)
