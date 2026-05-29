"""Version 1 乘法奖励: Total = r_task * exp(0.02 * r_aux)。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from train.version1.env import GenesisEnvV1

# 对角步态默认足端相位偏置 (rad): FR/RL 同相, FL/RR 同相
LEG_PHASE_OFFSETS = (0.0, math.pi, math.pi, 0.0)

# 动作镜像: FR(0:3)<->RL(9:12), FL(3:6)<->RR(6:9)；hip 关节取反
_MIRROR_SRC = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
_MIRROR_DST = (9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2)
_MIRROR_SIGN = torch.tensor(
    [-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0],
    dtype=torch.float32,
)


def mirror_actions(actions: torch.Tensor) -> torch.Tensor:
    """四足左右/前后对角镜像动作。"""
    sign = _MIRROR_SIGN.to(device=actions.device, dtype=actions.dtype)
    return actions[:, _MIRROR_DST] * sign


def _foot_contact_forces_z(env: GenesisEnvV1) -> torch.Tensor:
    """逐足聚合接触竖直力 [num_envs, 4]；无接触时为 0。"""
    n_feet = len(env.foot_link_indices)
    out = torch.zeros((env.num_envs, n_feet), dtype=torch.float32, device=env.device)
    contacts = env.robot.get_contacts()
    link_a = contacts["link_a"]
    link_b = contacts["link_b"]
    if link_a.dim() == 1:
        return out

    force_a = contacts["force_a"]
    force_b = contacts["force_b"]
    valid = contacts.get("valid_mask")

    n_envs = env.num_envs
    z = torch.zeros(n_envs, dtype=torch.float32, device=env.device)
    for foot_i, link_idx in enumerate(env.foot_link_indices):
        match_a = link_a == link_idx
        match_b = link_b == link_idx
        if valid is not None:
            match_a = match_a & valid
            match_b = match_b & valid
        fz = z.clone()
        if match_a.any():
            fa_z = torch.where(match_a, force_a[..., 2].abs(), torch.zeros_like(force_a[..., 2]))
            if fa_z.shape[1] > 0:
                fz = torch.max(fz, fa_z.max(dim=1)[0])
        if match_b.any():
            fb_z = torch.where(match_b, force_b[..., 2].abs(), torch.zeros_like(force_b[..., 2]))
            if fb_z.shape[1] > 0:
                fz = torch.max(fz, fb_z.max(dim=1)[0])
        out[:, foot_i] = fz
    return out


def compute_v1_rewards(env: GenesisEnvV1) -> torch.Tensor:
    cfg = env.cfg.reward
    cmd = env.velocity_commands
    vx_err = torch.square(env.base_lin_vel[:, 0] - cmd[:, 0])
    vy_err = torch.square(env.base_lin_vel[:, 1] - cmd[:, 1])
    wz_err = torch.square(env.base_ang_vel[:, 2] - cmd[:, 2])

    r_task = (
        cfg.task_weight_vx * torch.exp(-vx_err / cfg.task_sigma_vx)
        + cfg.task_weight_vy * torch.exp(-vy_err / cfg.task_sigma_vy)
        + cfg.task_weight_wz * torch.exp(-wz_err / cfg.task_sigma_wz)
    )

    l_action_rate = torch.sum(torch.square(env.actions - env.last_actions), dim=-1)
    torques = env.robot.get_dofs_control_force(env.motors_dof_idx)
    l_torque = torch.sum(torch.square(torques), dim=-1)
    l_dof_vel = torch.sum(torch.square(env.dof_vel), dim=-1)

    mirrored = mirror_actions(env.actions)
    l_sym = torch.sum(torch.square(env.actions - mirrored), dim=-1)

    l_phase = _phase_tracking_penalty(env)

    r_aux = (
        cfg.w_action_rate * l_action_rate
        + cfg.w_torque * l_torque
        + cfg.w_phase * l_phase
        + cfg.w_symmetry * l_sym
        + cfg.w_dof_vel * l_dof_vel
    )

    return r_task * torch.exp(cfg.aux_exp_scale * r_aux)


def _phase_tracking_penalty(env: GenesisEnvV1) -> torch.Tensor:
    """连续相位追踪惩罚（越小越好，权重为负）。"""
    freq = env.mob_commands[:, 0]
    phase_cmd = env.mob_commands[:, 2]
    t = env.episode_length_buf.to(dtype=torch.float32) * env.cfg.control_dt
    foot_fz = _foot_contact_forces_z(env)
    foot_vel_xy = env.foot_lin_vel[:, :, :2]
    foot_speed_xy = torch.linalg.norm(foot_vel_xy, dim=-1)

    penalty = torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)
    offsets = torch.tensor(LEG_PHASE_OFFSETS, dtype=torch.float32, device=env.device)

    for leg_i in range(4):
        phi = 2.0 * math.pi * freq * t + offsets[leg_i] + phase_cmd
        c_i = torch.sin(phi)
        fz = foot_fz[:, leg_i]
        vxy = foot_speed_xy[:, leg_i]

        swing_violation = torch.relu(c_i) * fz
        stance_violation = torch.relu(-c_i) * torch.abs(c_i) * vxy
        penalty = penalty + swing_violation + stance_violation

    return penalty


def sample_privileged_state(env: GenesisEnvV1, envs_idx: torch.Tensor | None) -> None:
    """Reset 时为 Critic 特权量采样域随机参数。"""
    cfg = env.cfg
    gen = getattr(env, "command_generator", None)

    if envs_idx is None:
        n = env.num_envs
        target = slice(None)
    else:
        n = envs_idx.numel()
        target = envs_idx

    lo_f, hi_f = cfg.domain_rand_friction_range
    env.privileged_friction[target, 0] = (hi_f - lo_f) * torch.rand(
        (n,), dtype=torch.float32, device=env.device, generator=gen
    ) + lo_f

    lo_c, hi_c = cfg.domain_rand_com_offset_range
    env.privileged_com_offset[target] = (hi_c - lo_c) * torch.rand(
        (n, 3), dtype=torch.float32, device=env.device, generator=gen
    ) + lo_c
