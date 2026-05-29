# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

gym.register(
    id='RobotLab-Isaac-Velocity-Flat-Taili-Quad-v0',
    entry_point='isaaclab.envs:ManagerBasedRLEnv',
    disable_env_checker=True,
    kwargs={
        'env_cfg_entry_point': f'{__name__}.flat_env_cfg:TailiQuadFlatEnvCfg',
        'rsl_rl_cfg_entry_point': f'{agents.__name__}.rsl_rl_ppo_cfg:TailiQuadFlatPPORunnerCfg',
        'cusrl_cfg_entry_point': f'{agents.__name__}.cusrl_ppo_cfg:TailiQuadFlatTrainerCfg',
    },
)

gym.register(
    id='RobotLab-Isaac-Velocity-Rough-Taili-Quad-v0',
    entry_point='isaaclab.envs:ManagerBasedRLEnv',
    disable_env_checker=True,
    kwargs={
        'env_cfg_entry_point': f'{__name__}.rough_env_cfg:TailiQuadRoughEnvCfg',
        'rsl_rl_cfg_entry_point': f'{agents.__name__}.rsl_rl_ppo_cfg:TailiQuadRoughPPORunnerCfg',
        'cusrl_cfg_entry_point': f'{agents.__name__}.cusrl_ppo_cfg:TailiQuadRoughTrainerCfg',
    },
)