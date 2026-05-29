"""Version 1 —— 平地盲跑与站立 (Flat Ground Velocity Tracking)。"""

from train import register_version
from train.version1.cfg import build_v1_env_cfg
from train.version1.env import GenesisEnvV1
from train.version1.train_cfg import build_v1_train_cfg

register_version(
    "version1",
    env_cls=GenesisEnvV1,
    build_env_cfg=build_v1_env_cfg,
    build_train_cfg=build_v1_train_cfg,
    default_exp_name="taili-v1-flat",
)

__all__ = ["GenesisEnvV1", "build_v1_env_cfg", "build_v1_train_cfg"]
