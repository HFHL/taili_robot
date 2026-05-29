"""随机种子 —— 训练可复现；采样在 gs.device 上执行（MPS / CUDA / CPU）。"""

from __future__ import annotations

import random

import torch

import genesis as gs


def set_global_seed(seed: int) -> None:
    """
    设置 Python / NumPy / PyTorch 全局种子。

    Genesis 随机性主要由 ``gs.init(..., seed=seed)`` 控制；此处同步 PyTorch 侧，
    便于指令重采样等与 Genesis 无关的张量随机操作。
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env_generator(seed: int, env_index: int) -> torch.Generator:
    """为单个并行环境派生独立 Generator（仍落在 gs.device 上）。"""
    gen = torch.Generator(device=gs.device)
    gen.manual_seed(seed + env_index * 100003)
    return gen
