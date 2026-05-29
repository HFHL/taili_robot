"""RL 策略版本注册 —— 每个版本在 train/<version>/ 下独立维护。"""

from __future__ import annotations

from typing import Callable, Type

from envs.base_env import BaseEnv

VERSIONS: dict[str, dict[str, object]] = {}


def register_version(
    name: str,
    *,
    env_cls: Type[BaseEnv],
    build_env_cfg: Callable[..., object],
    build_train_cfg: Callable[..., dict],
    default_exp_name: str,
) -> None:
    VERSIONS[name] = {
        "env_cls": env_cls,
        "build_env_cfg": build_env_cfg,
        "build_train_cfg": build_train_cfg,
        "default_exp_name": default_exp_name,
    }


def get_version(name: str) -> dict[str, object]:
    if name not in VERSIONS:
        known = ", ".join(sorted(VERSIONS)) or "(none loaded)"
        raise KeyError(f"未知训练版本 '{name}'，已注册: {known}")
    return VERSIONS[name]


def list_versions() -> list[str]:
    return sorted(VERSIONS)


def _load_version_modules() -> None:
    """导入各版本包以完成注册（新增版本时在此追加）。"""
    import train.version1  # noqa: F401


_load_version_modules()
