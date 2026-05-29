"""训练入口 —— 按 RL 策略版本分发到 train/<version>/。"""

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
from train import get_version, list_versions


def build_legacy_env_cfg(num_envs: int, show_viewer: bool = False, seed: int = 42) -> EnvCfg:
    """旧版基线配置（无版本目录时使用）。"""
    return EnvCfg(
        sim=SimCfg(num_envs=num_envs, sim_dt=0.02, substeps=2, episode_length_s=20.0, seed=seed),
        robot=RobotCfg(),
        action=ActionCfg(),
        obs=ObsCfg(),
        command=CommandCfg(),
        reward=RewardCfg(),
        termination=TerminationCfg(),
        show_viewer=show_viewer,
    )


def build_legacy_train_cfg(exp_name: str, max_iterations: int) -> dict:
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
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
        "max_iterations": max_iterations,
        "train_version": "legacy",
    }


def _run_training(
    *,
    version: str,
    exp_name: str,
    num_envs: int,
    max_iterations: int,
    seed: int,
    show_viewer: bool,
    use_cpu: bool,
) -> None:
    if version == "legacy":
        env_cfg = build_legacy_env_cfg(num_envs, show_viewer=show_viewer, seed=seed)
        train_cfg = build_legacy_train_cfg(exp_name, max_iterations)
        env_cls = GenesisEnv
        train_version = "legacy"
    else:
        spec = get_version(version)
        env_cfg = spec["build_env_cfg"](num_envs=num_envs, show_viewer=show_viewer, seed=seed)
        train_cfg = spec["build_train_cfg"](exp_name, max_iterations)
        env_cls = spec["env_cls"]
        train_version = version
        if not exp_name or exp_name == "taili-locomotion":
            exp_name = str(spec["default_exp_name"])

    run_dir, run_id = make_run_dir(exp_name)
    backend = gs.cpu if use_cpu else gs.gpu
    genesis_init = {
        "backend": "cpu" if use_cpu else "gpu",
        "precision": "32",
        "logging_level": "warning",
        "seed": seed,
        "performance_mode": True,
    }

    run_metadata = build_run_metadata(
        exp_name=exp_name,
        run_id=run_id,
        run_dir=run_dir,
        cli_args={
            "train_version": train_version,
            "exp_name": exp_name,
            "num_envs": num_envs,
            "max_iterations": max_iterations,
            "seed": seed,
            "show_viewer": show_viewer,
            "cpu": use_cpu,
        },
        env_cfg=env_cfg,
        train_cfg=train_cfg,
        genesis_init=genesis_init,
    )
    write_run_manifest(run_dir, run_metadata)
    started_at_utc = run_metadata["timestamps"]["started_at_utc"]

    with open(run_dir / "cfgs.pkl", "wb") as f:
        pickle.dump({"env_cfg": env_cfg, "train_cfg": train_cfg, "train_version": train_version}, f)

    print(f"[train] 版本: {train_version}")
    print(f"[run] 配置: {run_dir / 'config.txt'}")
    print(f"[run] 目录: {run_dir}")

    gs.init(
        backend=backend,
        precision=genesis_init["precision"],
        logging_level=genesis_init["logging_level"],
        seed=seed,
        performance_mode=genesis_init["performance_mode"],
    )
    set_global_seed(seed)

    env = env_cls(cfg=env_cfg)
    write_pre_train_log(
        run_dir,
        env,
        extra={
            "train_version": train_version,
            "genesis_backend": genesis_init["backend"],
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
            runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)
            status = "completed"
            iterations_completed = max_iterations

        except (ImportError, pkg_metadata.PackageNotFoundError):
            status = "env_only"
            print("[WARN] 未检测到 rsl-rl-lib>=5.0.0，跳过训练循环。")
            print(f"环境已就绪: obs={getattr(env, 'obs_buf', None)}")

        except Exception as exc:
            status = "failed"
            error_msg = f"{type(exc).__name__}: {exc}"
            raise

        finally:
            write_post_train_summary(
                run_dir,
                status=status,
                started_at_utc=started_at_utc,
                max_iterations=max_iterations,
                iterations_completed=iterations_completed,
                error=error_msg,
            )


def main() -> None:
    versions = list_versions()
    parser = argparse.ArgumentParser(description="Genesis RL 训练（按版本分发）")
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="version1",
        choices=versions + ["legacy"],
        help=f"策略版本目录 train/<version>/，已注册: {', '.join(versions)}",
    )
    parser.add_argument("-e", "--exp_name", type=str, default="")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show_viewer", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    _run_training(
        version=args.version,
        exp_name=args.exp_name,
        num_envs=args.num_envs,
        max_iterations=args.max_iterations,
        seed=args.seed,
        show_viewer=args.show_viewer,
        use_cpu=args.cpu,
    )


if __name__ == "__main__":
    main()
