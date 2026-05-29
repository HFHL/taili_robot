#!/usr/bin/env python3
"""
验证张量化部分重置：仅 reset 指定 env，其它 env 的 episode_length 连续递增。

在 genesis-world 根目录:
  uv run python taili/scripts/check_vectorized_reset.py
  uv run python taili/scripts/check_vectorized_reset.py --cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import genesis as gs

from configs.env_cfg import EnvCfg, SimCfg
from envs.genesis_env import GenesisEnv


def main() -> int:
    parser = argparse.ArgumentParser(description="检查部分环境重置")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("-B", "--num_envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend, seed=args.seed, logging_level="warning")

    cfg = EnvCfg(sim=SimCfg(num_envs=args.num_envs, seed=args.seed))
    env = GenesisEnv(cfg=cfg)

    assert str(env.device) == str(gs.device), f"device 不一致: {env.device} vs {gs.device}"
    assert env.obs_buf.device == env.device
    assert env.actions.device == env.device

    # 人为让 env 0 立即超时（改短 max episode）
    env.max_episode_length = 3

    actions = torch.zeros((args.num_envs, env.num_actions), device=env.device)
    lengths_before: list[torch.Tensor] = []

    for step in range(6):
        lengths_before.append(env.episode_length_buf.clone())
        _, _, dones, _ = env.step(actions)
        print(
            f"step {step + 1} | episode_length={env.episode_length_buf.tolist()} "
            f"| dones={dones.tolist()}"
        )

    # env0 应在某步 done 后 length 归零；其它 env 应持续增加（未被人为 reset）
    l5 = lengths_before[5]
    assert env.episode_length_buf[0].item() <= 3, "env0 应已 reset 后重新计数"
    assert env.episode_length_buf[1].item() >= 5, "env1 应持续步进，不受 env0 reset 影响"

    # 仅 reset env 1，检查 env 2 的 episode_length 不变
    ref_len = env.episode_length_buf[2].item()
    env.reset_idx(torch.tensor([1], device=env.device))
    assert env.episode_length_buf[1].item() == 0
    assert env.episode_length_buf[2].item() == ref_len, "reset_idx(1) 不应影响 env2 步数"

    print("PASS: 张量化部分重置与 device 检查通过。")
    print(f"  device={env.device} (Apple Silicon 上 gpu 一般为 mps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
