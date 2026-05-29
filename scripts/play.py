#!/usr/bin/env python3
"""
阶段四：策略可视化 (Play / Inference)

从训练 run 目录加载 cfgs.pkl + model_*.pt，在 Genesis Viewer 中运行策略。

示例（genesis-world 根目录）::

    uv run python taili/scripts/play.py -e taili-baseline --ckpt 499
    uv run python taili/scripts/play.py --run_dir taili/logs/taili-baseline/20260529_165616
"""

from __future__ import annotations

import argparse
import pickle
import sys
from importlib import metadata as pkg_metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import genesis as gs
import torch

from configs.run_manifest import find_checkpoint, resolve_run_dir
from envs.genesis_env import GenesisEnv


def main() -> int:
    parser = argparse.ArgumentParser(description="加载 checkpoint 并在 Viewer 中运行策略")
    parser.add_argument("-e", "--exp_name", type=str, default="taili-baseline")
    parser.add_argument("--run_id", type=str, default=None, help="YYYYMMDD_HHMMSS，默认最新 completed run")
    parser.add_argument("--run_dir", type=str, default=None, help="直接指定 run 目录（优先于 exp_name/run_id）")
    parser.add_argument("--ckpt", type=int, default=None, help="checkpoint iteration，默认最大 model_*.pt")
    parser.add_argument("-B", "--num_envs", type=int, default=1, help="并行环境数（Viewer 建议 1）")
    parser.add_argument("--seed", type=int, default=None, help="覆盖训练 seed（默认读 cfgs.pkl）")
    parser.add_argument("--max_steps", type=int, default=0, help="最大步数，0 表示无限循环")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--no-viewer", action="store_true", help="无头模式（仅验证加载与推理）")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    try:
        if int(pkg_metadata.version("rsl-rl-lib").split(".")[0]) < 5:
            raise ImportError
        from rsl_rl.runners import OnPolicyRunner
    except (ImportError, pkg_metadata.PackageNotFoundError):
        print("[ERROR] 需要 rsl-rl-lib>=5.0.0: pip install rsl-rl-lib>=5.0.0")
        return 1

    run_dir = resolve_run_dir(args.exp_name, run_id=args.run_id, run_dir=args.run_dir)
    ckpt_path = find_checkpoint(run_dir, args.ckpt)
    print(f"[play] run_dir:  {run_dir}")
    print(f"[play] checkpoint: {ckpt_path.name}")

    with open(run_dir / "cfgs.pkl", "rb") as f:
        saved = pickle.load(f)
    env_cfg = saved["env_cfg"]
    train_cfg = saved["train_cfg"]

    env_cfg.sim.num_envs = args.num_envs
    env_cfg.show_viewer = not args.no_viewer
    seed = args.seed if args.seed is not None else env_cfg.sim.seed

    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend, seed=seed, logging_level="warning", performance_mode=True)

    env = GenesisEnv(cfg=env_cfg)
    runner = OnPolicyRunner(env, train_cfg, str(run_dir), device=gs.device)
    runner.load(str(ckpt_path))
    policy = runner.get_inference_policy(device=gs.device)

    obs = env.reset()
    step = 0
    print(f"[play] num_envs={env.num_envs} device={env.device} viewer={not args.no_viewer}")
    print("[play] Ctrl+C 退出")

    try:
        with torch.inference_mode():
            while args.max_steps == 0 or step < args.max_steps:
                actions = policy(obs)
                obs, rew, dones, extras = env.step(actions)
                step += 1
                if step == 1 or step % args.log_interval == 0:
                    ep_len = env.episode_length_buf.float().mean().item()
                    n_reset = extras.get("n_reset_envs", 0)
                    print(
                        f"step {step:6d} | mean_rew={rew.mean().item():.3f} "
                        f"| mean_ep_len={ep_len:.1f} | dones={dones.sum().item():.0f} "
                        f"| n_reset={n_reset}"
                    )
    except KeyboardInterrupt:
        print(f"\n[play] 停止于 step {step}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
