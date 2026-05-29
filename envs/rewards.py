"""奖励函数模块 —— 与 Env 解耦，通过 Mixin 注入。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict

import torch

if TYPE_CHECKING:
    from envs.genesis_env import GenesisEnv


class RewardMixin:
    """
    奖励 Mixin：各奖励项为 _reward_<name>，权重来自 cfg.reward.reward_weights。

    基础阶段默认:
        alive: 未终止时每步 +1
        tracking_lin_vel / tracking_ang_vel: 接近速度指令
    """

    cfg: "GenesisEnv 需持有 EnvCfg"  # type: ignore[name-defined]
    num_envs: int
    device: torch.device

    actions: torch.Tensor
    last_actions: torch.Tensor
    dof_pos: torch.Tensor
    default_dof_pos: torch.Tensor
    base_lin_vel: torch.Tensor
    base_ang_vel: torch.Tensor
    commands: torch.Tensor
    episode_length_buf: torch.Tensor
    reset_buf: torch.Tensor

    def _build_reward_registry(self) -> None:
        self._reward_funcs: Dict[str, Callable[[], torch.Tensor]] = {}
        self._reward_weights: Dict[str, float] = {}
        self._episode_reward_sums: Dict[str, torch.Tensor] = {}

        dt = self.cfg.sim.sim_dt
        for name, weight in self.cfg.reward.reward_weights.items():
            if weight == 0.0:
                continue
            func_name = f"_reward_{name}"
            if not hasattr(self, func_name):
                raise AttributeError(
                    f"奖励项 '{name}' 已配置，但未实现 '{func_name}'。"
                )
            self._reward_funcs[name] = getattr(self, func_name)
            self._reward_weights[name] = weight * dt
            self._episode_reward_sums[name] = torch.zeros(
                (self.num_envs,), dtype=torch.float32, device=self.device
            )

    def compute_rewards(self) -> torch.Tensor:
        total_reward = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        for name, func in self._reward_funcs.items():
            rew = func()
            weighted_rew = rew * self._reward_weights[name]
            total_reward += weighted_rew
            self._episode_reward_sums[name] += weighted_rew
        return total_reward

    def _reset_episode_reward_sums(self, env_ids: torch.Tensor | None = None) -> None:
        for buf in self._episode_reward_sums.values():
            if env_ids is None:
                buf.zero_()
            elif env_ids.dtype == torch.bool:
                buf.masked_fill_(env_ids, 0.0)
            else:
                buf[env_ids] = 0.0

    # ===================== 基础奖励 =====================

    def _reward_alive(self) -> torch.Tensor:
        """存活：本步未触发终止的环境为 1.0。"""
        return (~self.reset_buf).float()

    def _reward_tracking_lin_vel(self) -> torch.Tensor:
        """跟踪线速度指令 (vx, vy)，机体系。"""
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]),
            dim=-1,
        )
        return torch.exp(-lin_vel_error / self.cfg.reward.tracking_sigma)

    def _reward_tracking_ang_vel(self) -> torch.Tensor:
        """跟踪偏航角速度指令 (yaw rate)。"""
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.reward.tracking_sigma)
