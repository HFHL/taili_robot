#!/usr/bin/env python3
"""
阶段四：训练 run 诊断 —— 解析 train.log，输出指标趋势与改进建议。

示例::

    uv run python taili/scripts/analyze_run.py -e taili-baseline
    uv run python taili/scripts/analyze_run.py --run_dir taili/logs/taili-baseline/20260529_165616
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.run_manifest import resolve_run_dir


def _parse_train_log(log_path: Path) -> dict[int, dict[str, float]]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"Learning iteration (\d+)/(\d+)", text)
    max_it = 0
    iters: dict[int, dict[str, float]] = {}

    def grab(body: str, key: str) -> float | None:
        m = re.search(rf"{re.escape(key)}:\s*([-\d.]+)", body)
        return float(m.group(1)) if m else None

    for i in range(1, len(blocks), 3):
        it = int(blocks[i])
        total = int(blocks[i + 1])
        max_it = max(max_it, total)
        body = blocks[i + 2]
        iters[it] = {
            "value_loss": grab(body, "Mean value loss"),
            "entropy_loss": grab(body, "Mean entropy loss"),
            "mean_reward": grab(body, "Mean reward"),
            "episode_length": grab(body, "Mean episode length"),
            "action_std": grab(body, "Mean action std"),
        }
    return iters, max_it


def _pct_change(a: float | None, b: float | None) -> str:
    if a is None or b is None or a == 0:
        return "N/A"
    return f"{100.0 * (b - a) / abs(a):+.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description="诊断训练 run 指标趋势")
    parser.add_argument("-e", "--exp_name", type=str, default="taili-baseline")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--run_dir", type=str, default=None)
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.exp_name, run_id=args.run_id, run_dir=args.run_dir)
    post_path = run_dir / "post_train.json"
    log_path = run_dir / "train.log"

    print("=" * 72)
    print(f"Run: {run_dir}")
    if post_path.is_file():
        post = json.loads(post_path.read_text(encoding="utf-8"))
        print(f"状态: {post.get('status')} | 耗时: {post.get('timestamps', {}).get('duration_s', 0):.0f}s")
        tr = post.get("training", {})
        print(f"迭代: {tr.get('iterations_completed')} / {tr.get('max_iterations_requested')}")
        print(f"检查点: {', '.join(post.get('artifacts', {}).get('checkpoints', []))}")

    if not log_path.is_file():
        print("\n[WARN] 无 train.log，跳过指标解析")
        return 0

    iters, max_it = _parse_train_log(log_path)
    if not iters:
        print("\n[WARN] train.log 中未解析到 iteration 块")
        return 0

    last_it = max(iters)
    first = iters[min(iters)]
    last = iters[last_it]
    mid_it = max(k for k in iters if k <= max_it // 2)
    mid = iters[mid_it]

    print("\n--- 指标趋势 (iter 0 → last) ---")
    rows = [
        ("Mean reward", "mean_reward"),
        ("Mean episode length", "episode_length"),
        ("Value loss", "value_loss"),
        ("Entropy loss", "entropy_loss"),
        ("Action std", "action_std"),
    ]
    for label, key in rows:
        print(
            f"  {label:22s}  {first.get(key)!s:>8} → {last.get(key)!s:>8}  "
            f"({_pct_change(first.get(key), last.get(key))})"
        )

    print("\n--- 阶段四诊断建议 ---")
    suggestions: list[str] = []

    ep_len = last.get("episode_length")
    max_ep = None
    cfg_txt = run_dir / "config.txt"
    if cfg_txt.is_file() and "max_episode_length:" in cfg_txt.read_text(encoding="utf-8"):
        m = re.search(r"max_episode_length:\s*(\d+)", cfg_txt.read_text(encoding="utf-8"))
        if m:
            max_ep = int(m.group(1))

    if ep_len is not None and max_ep and ep_len < 0.15 * max_ep:
        suggestions.append(
            f"Episode 偏短（{ep_len:.0f}/{max_ep} 步）：检查终止条件是否过严，或 PD/初始姿态导致频繁倒地。"
        )

    rew0, rew1 = first.get("mean_reward"), last.get("mean_reward")
    if rew0 is not None and rew1 is not None and rew1 <= rew0 * 1.1:
        suggestions.append("Mean reward 几乎无增长：检查 CommandCfg 指令范围、奖励权重、观测是否含 commands。")

    ent0, ent1 = first.get("entropy_loss"), last.get("entropy_loss")
    if ent0 is not None and ent1 is not None and ent1 >= ent0:
        suggestions.append(
            "Entropy 未下降：策略仍较随机；可延长训练，或略降 entropy_coef / 检查 action_std 是否过大。"
        )

    std1 = last.get("action_std")
    if std1 is not None and std1 > 0.8:
        suggestions.append(
            "Action std 仍高：在 Viewer 中运行 play.py 观察是否抽搐；可考虑 action_rate 惩罚或减小 action_scale。"
        )

    if mid.get("mean_reward") and last.get("mean_reward"):
        if last["mean_reward"] <= mid["mean_reward"] * 1.05:
            suggestions.append("后半段 reward 平台化：尝试调 tracking 权重、learning_rate，或增加 num_envs。")

    if not suggestions:
        suggestions.append("指标整体正常；请用 play.py 目视行为，再决定是否微调奖励。")

    for i, s in enumerate(suggestions, 1):
        print(f"  {i}. {s}")

    print("\n--- 下一步 ---")
    print(f"  目视策略: uv run python taili/scripts/play.py --run_dir {run_dir}")
    print(f"  TensorBoard: tensorboard --logdir {run_dir}")
    print(f"  对比配置: diff logs/<exp>/*/config.txt")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
