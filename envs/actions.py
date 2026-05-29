"""
动作空间与控制映射 —— 策略输出到 Genesis PD 目标关节角。

动作空间（taili_quad 默认）::

    形状: [num_envs, num_actions]，num_actions = 12
    语义: 每个元素对应 ``RobotCfg.joint_names`` 中的一条腿关节（策略侧无单位，建议 ∈ [-1, 1]）

控制映射（位置 PD，与 Go2 / Isaac Lab 一致）::

    1. a = clip(a_raw, -clip_actions, +clip_actions)
    2. a_exec = last_actions  （若启用 1 步动作延迟）
    3. q_target = q_default + a_exec ⊙ action_scale   （⊙ 为逐关节缩放，可统一或逐关节）
    4. q_target = clamp(q_target, 软限位)              （URDF 限位 × soft_joint_pos_limit_factor）
    5. robot.control_dofs_position(q_target, motors_dof_idx)  （Genesis 内部 PD 跟踪）
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import torch

if TYPE_CHECKING:
    from configs.env_cfg import ActionCfg


def action_spec(joint_names: List[str], cfg: ActionCfg) -> dict:
    """动作空间元信息，写入 env.extras。"""
    return {
        "shape": [len(joint_names)],
        "joint_names": list(joint_names),
        "clip_actions": cfg.clip_actions,
        "action_scale": cfg.action_scale,
        "control_mode": cfg.control_mode,
        "mapping": "q_target = q_default + clip(action) * action_scale (per-joint scale supported)",
    }


def build_action_scale(
    cfg: ActionCfg,
    joint_names: List[str],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """构建 [num_actions] 逐关节缩放向量。"""
    n = len(joint_names)
    if cfg.action_scale_per_joint:
        return torch.tensor(
            [cfg.action_scale_per_joint.get(name, cfg.action_scale) for name in joint_names],
            dtype=dtype,
            device=device,
        )
    return torch.full((n,), cfg.action_scale, dtype=dtype, device=device)


def clip_policy_actions(raw_actions: torch.Tensor, cfg: ActionCfg) -> torch.Tensor:
    """策略输出裁剪到 [-clip_actions, clip_actions]。"""
    return torch.clamp(raw_actions, -cfg.clip_actions, cfg.clip_actions)


def select_executed_actions(
    actions: torch.Tensor,
    last_actions: torch.Tensor,
    cfg: ActionCfg,
) -> torch.Tensor:
    """可选 1 步动作延迟：执行上一帧动作。"""
    if cfg.simulate_action_latency:
        return last_actions
    return actions


def actions_to_target_dof_pos(
    exec_actions: torch.Tensor,
    default_dof_pos: torch.Tensor,
    action_scale: torch.Tensor,
    dof_pos_lower: torch.Tensor,
    dof_pos_upper: torch.Tensor,
    soft_joint_pos_limit_factor: float,
) -> torch.Tensor:
    """
    将归一化动作映射为目标关节角 [num_envs, num_actions]。

    Parameters
    ----------
    exec_actions : [num_envs, num_actions]
    default_dof_pos : [num_actions] 或 [num_envs, num_actions]
    action_scale : [num_actions]
    dof_pos_lower, dof_pos_upper : [num_actions]，URDF 硬限位
    """
    if default_dof_pos.dim() == 1:
        default_dof_pos = default_dof_pos.unsqueeze(0)

    target = default_dof_pos + exec_actions * action_scale.unsqueeze(0)

    mid = 0.5 * (dof_pos_lower + dof_pos_upper)
    half_span = 0.5 * (dof_pos_upper - dof_pos_lower) * soft_joint_pos_limit_factor
    return torch.clamp(target, mid - half_span, mid + half_span)


def normalize_dof_limits(
    lower: torch.Tensor,
    upper: torch.Tensor,
    num_actions: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """将 Genesis get_dofs_limit 返回值规整为 [num_actions]。"""
    lo = lower.reshape(-1, num_actions)
    hi = upper.reshape(-1, num_actions)
    if lo.shape[0] > 1:
        lo = lo[0]
        hi = hi[0]
    else:
        lo = lo.squeeze(0)
        hi = hi.squeeze(0)
    return lo, hi
