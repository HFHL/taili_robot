"""Version 1 训练入口。"""

from __future__ import annotations

import argparse
import pickle
import sys
from importlib import metadata as pkg_metadata
from pathlib import Path

TAILI_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TAILI_ROOT))

import genesis as gs

from configs.run_manifest import (
    build_run_metadata,
    make_run_dir,
    tee_train_log,
    write_post_train_summary,
    write_pre_train_log,
    write_run_manifest,
)
from envs.seeding import set_global_seed
from train.version1.cfg import VERSION_LABEL, VERSION_NAME, build_v1_env_cfg
from train.version1.env import GenesisEnvV1
from train.version1.train_cfg import build_v1_train_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Taili RL v1 — 平地盲跑与站立")
    parser.add_argument("-e", "--exp_name", type=str, default="taili-v1-flat")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show_viewer", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    run_dir, run_id = make_run_dir(args.exp_name)
    env_cfg = build_v1_env_cfg(num_envs=args.num_envs, show_viewer=args.show_viewer, seed=args.seed)
    train_cfg = build_v1_train_cfg(args.exp_name, args.max_iterations)

    backend = gs.cpu if args.cpu else gs.gpu
    genesis_init = {
        "backend": "cpu" if args.cpu else "gpu",
        "precision": "32",
        "logging_level": "warning",
        "seed": args.seed,
        "performance_mode": True,
    }

    run_metadata = build_run_metadata(
        exp_name=args.exp_name,
        run_id=run_id,
        run_dir=run_dir,
        cli_args={**vars(args), "train_version": VERSION_NAME},
        env_cfg=env_cfg,
        train_cfg=train_cfg,
        genesis_init=genesis_init,
    )
    write_run_manifest(run_dir, run_metadata)
    started_at_utc = run_metadata["timestamps"]["started_at_utc"]

    with open(run_dir / "cfgs.pkl", "wb") as f:
        pickle.dump({"env_cfg": env_cfg, "train_cfg": train_cfg, "train_version": VERSION_NAME}, f)

    print(f"[v1] {VERSION_LABEL}")
    print(f"[run] 配置: {run_dir / 'config.txt'}")
    print(f"[run] 目录: {run_dir}")

    gs.init(
        backend=backend,
        precision=genesis_init["precision"],
        logging_level=genesis_init["logging_level"],
        seed=args.seed,
        performance_mode=genesis_init["performance_mode"],
    )
    set_global_seed(args.seed)

    env = GenesisEnvV1(cfg=env_cfg)
    write_pre_train_log(
        run_dir,
        env,
        extra={
            "train_version": VERSION_NAME,
            "genesis_backend": genesis_init["backend"],
            "num_critic_obs": env.cfg.num_critic_obs,
            "control_dt": env.cfg.control_dt,
            "reward_structure": "r_task * exp(0.02 * r_aux)",
        },
    )

    status = "skipped"
    iterations_completed: int | None = None
    error_msg: str | None = None

    with tee_train_log(run_dir):
        try:
            if int(pkg_metadata.version("rsl-rl-lib").split(".")[0]) < 5:
                raise ImportError
            from rsl_rl.runners import OnPolicyRunner

            runner = OnPolicyRunner(env, train_cfg, str(run_dir), device=gs.device)
            runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
            status = "completed"
            iterations_completed = args.max_iterations

        except (ImportError, pkg_metadata.PackageNotFoundError):
            status = "env_only"
            print(
                "[WARN] 未检测到 rsl-rl-lib>=5.0.0，跳过训练循环。\n"
                "       请安装: pip install rsl-rl-lib>=5.0.0"
            )
            print(f"环境: policy_obs={env.obs_buf.shape} critic_obs={env.critic_obs_buf.shape}")

        except Exception as exc:
            status = "failed"
            error_msg = f"{type(exc).__name__}: {exc}"
            raise

        finally:
            write_post_train_summary(
                run_dir,
                status=status,
                started_at_utc=started_at_utc,
                max_iterations=args.max_iterations,
                iterations_completed=iterations_completed,
                error=error_msg,
            )


if __name__ == "__main__":
    main()
