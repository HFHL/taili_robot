"""训练入口 —— 极简骨架，展示 Env 实例化与 RL Runner 对接流程。"""

from __future__ import annotations

import argparse
import pickle
import sys
from importlib import metadata as pkg_metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import genesis as gs

from configs.env_cfg import (
    ActionCfg,
    CommandCfg,
    EnvCfg,
    ObsCfg,
    RewardCfg,
    RobotCfg,
    SimCfg,
    TerminationCfg,
)
from configs.run_manifest import (
    build_run_metadata,
    make_run_dir,
    tee_train_log,
    write_post_train_summary,
    write_pre_train_log,
    write_run_manifest,
)
from envs.genesis_env import GenesisEnv
from envs.seeding import set_global_seed


def build_env_cfg(num_envs: int, show_viewer: bool = False, seed: int = 42) -> EnvCfg:
    """组装环境配置；机器人 / 观测默认值见 configs/env_cfg.py。"""
    return EnvCfg(
        sim=SimCfg(
            num_envs=num_envs,
            sim_dt=0.02,
            substeps=2,
            episode_length_s=20.0,
            seed=seed,
        ),
        robot=RobotCfg(),
        action=ActionCfg(),
        obs=ObsCfg(),
        command=CommandCfg(),
        reward=RewardCfg(),
        termination=TerminationCfg(),
        show_viewer=show_viewer,
    )


def build_train_cfg(exp_name: str, max_iterations: int) -> dict:
    """组装 rsl_rl 训练配置。"""
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 1e-3,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
        "max_iterations": max_iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Genesis RL 训练入口")
    parser.add_argument("-e", "--exp_name", type=str, default="taili-locomotion")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show_viewer", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="使用 CPU 后端（调试用）")
    args = parser.parse_args()

    run_dir, run_id = make_run_dir(args.exp_name)

    env_cfg = build_env_cfg(
        num_envs=args.num_envs, show_viewer=args.show_viewer, seed=args.seed
    )
    train_cfg = build_train_cfg(args.exp_name, args.max_iterations)

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
        cli_args=vars(args),
        env_cfg=env_cfg,
        train_cfg=train_cfg,
        genesis_init=genesis_init,
    )
    write_run_manifest(run_dir, run_metadata)
    started_at_utc = run_metadata["timestamps"]["started_at_utc"]

    with open(run_dir / "cfgs.pkl", "wb") as f:
        pickle.dump({"env_cfg": env_cfg, "train_cfg": train_cfg}, f)

    print(f"[run] 配置已写入: {run_dir / 'config.txt'}")
    print(f"[run] 运行目录:   {run_dir}")

    gs.init(
        backend=backend,
        precision=genesis_init["precision"],
        logging_level=genesis_init["logging_level"],
        seed=args.seed,
        performance_mode=genesis_init["performance_mode"],
    )
    set_global_seed(args.seed)

    env = GenesisEnv(cfg=env_cfg)
    write_pre_train_log(
        run_dir,
        env,
        extra={
            "genesis_backend": genesis_init["backend"],
            "reward_weights_effective": {
                k: v * env_cfg.sim.sim_dt
                for k, v in env_cfg.reward.reward_weights.items()
            },
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
            runner.learn(
                num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
            )
            status = "completed"
            iterations_completed = args.max_iterations

        except (ImportError, pkg_metadata.PackageNotFoundError):
            status = "env_only"
            print(
                "[WARN] 未检测到 rsl-rl-lib>=5.0.0，跳过训练循环。\n"
                "       请安装: pip install rsl-rl-lib>=5.0.0"
            )
            print(f"环境已就绪: num_envs={env.num_envs}, num_actions={env.num_actions}")
            print(f"观测形状: {env.obs_buf.shape}  # [num_envs, num_obs]")
            print(f"观测分量: {env.extras.get('obs_components')}")
            print(f"设备: {env.device}")

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
            print(f"[run] 训练摘要: {run_dir / 'post_train.txt'}")


if __name__ == "__main__":
    main()
