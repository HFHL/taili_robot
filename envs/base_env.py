"""RL 环境抽象基类 —— 定义与算法库（rsl_rl / CleanRL）对接的最小接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch


class BaseEnv(ABC):
    """
    向量环境基类。

    约定接口与 rsl_rl OnPolicyRunner 兼容：
        obs, rew, done, extras = env.step(actions)
        obs = env.reset()

    所有返回的张量均在 GPU 上，batch 维为 num_envs。
    """

    num_envs: int
    num_actions: int
    device: torch.device

    @abstractmethod
    def reset(self) -> Any:
        """
        重置所有环境。

        Returns
        -------
        observations
            形状取决于具体实现，通常为 TensorDict 或 [num_envs, num_obs]。
        """

    @abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[Any, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        执行一步仿真。

        Parameters
        ----------
        actions : torch.Tensor
            形状 [num_envs, num_actions]，策略网络输出的原始动作。

        Returns
        -------
        observations
            下一步观测。
        rewards : torch.Tensor
            形状 [num_envs]，逐步奖励。
        dones : torch.Tensor
            形状 [num_envs]，bool 或 0/1，表示 episode 是否结束。
        extras : dict
            额外信息，如 time_outs、episode 统计等。
        """

    @abstractmethod
    def get_observations(self) -> Any:
        """返回当前观测（不步进仿真）。"""
