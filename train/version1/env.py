"""Version 1 Genesis 向量环境。"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from tensordict import TensorDict

import genesis as gs

from envs.genesis_env import GenesisEnv
from train.version1.cfg import V1EnvCfg
from train.version1.commands import resample_commands_v1
from train.version1.observations import (
    clip_observations,
    critic_obs_component_names,
    policy_obs_component_names,
    write_critic_observations,
    write_policy_observations,
)
from train.version1.rewards import compute_v1_rewards, sample_privileged_state
# Genesis 加载的 URDF 未单独解析 foot link，用 calf 连杆作足端接触代理
FOOT_PROXY_LINK_NAMES = ("FR_calf_Link", "FL_calf_Link", "RR_calf_Link", "RL_calf_Link")


class GenesisEnvV1(GenesisEnv):
    """
    平地盲跑 v1：48 维 Actor 观测、特权 Critic、乘法奖励、控制 decimation。
    """

    cfg: V1EnvCfg

    def __init__(self, cfg: V1EnvCfg) -> None:
        self.cfg = cfg
        self.dt = cfg.control_dt
        self.max_episode_length = cfg.max_episode_length
        self.num_envs = cfg.sim.num_envs
        self.num_actions = cfg.num_actions
        self.num_dofs = self.num_actions
        self.device = gs.device

        from envs.seeding import make_env_generator, set_global_seed

        if self.cfg.sim.seed is not None:
            set_global_seed(self.cfg.sim.seed)
        self.command_generator = make_env_generator(
            self.cfg.sim.seed if self.cfg.sim.seed is not None else 0,
            env_index=0,
        )

        self._build_scene()
        self._setup_robot()
        self._setup_foot_links()
        self._init_buffers()
        self.reset()

    def _build_scene(self) -> None:
        super()._build_scene()

    def _setup_robot(self) -> None:
        robot_cfg = self.cfg.robot
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=self._resolve_path(robot_cfg.urdf_path),
                pos=robot_cfg.base_init_pos,
                quat=robot_cfg.base_init_quat,
            ),
        )
        self.scene.build(n_envs=self.num_envs, env_spacing=self.cfg.sim.env_spacing)

        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in robot_cfg.joint_names],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        self.robot.set_dofs_kp([robot_cfg.kp] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([robot_cfg.kd] * self.num_actions, self.motors_dof_idx)

        self.default_dof_pos = torch.tensor(
            [robot_cfg.default_joint_angles[name] for name in robot_cfg.joint_names],
            dtype=gs.tc_float,
            device=gs.device,
        )

        from genesis.utils.geom import inv_quat, transform_by_quat

        self.init_base_pos = torch.tensor(robot_cfg.base_init_pos, dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(robot_cfg.base_init_quat, dtype=gs.tc_float, device=gs.device)
        self.init_dof_pos = self.default_dof_pos.clone()
        self.init_qpos = self.robot.get_qpos().clone()
        if self.init_qpos.dim() == 1:
            self.init_qpos = self.init_qpos.unsqueeze(0).expand(self.num_envs, -1).contiguous()

        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)
        self._setup_action_limits()

    def _resolve_path(self, path: str) -> str:
        from configs.paths import resolve_asset_path

        return resolve_asset_path(path)

    def _setup_foot_links(self) -> None:
        self.foot_link_indices = [self.robot.get_link(name).idx for name in FOOT_PROXY_LINK_NAMES]
        self.foot_links_idx_local = torch.tensor(
            [self.robot.get_link(name).idx_local for name in FOOT_PROXY_LINK_NAMES],
            dtype=gs.tc_int,
            device=gs.device,
        )

    def _init_buffers(self) -> None:
        n = self.num_envs
        dev = gs.device
        f = gs.tc_float

        self.actions = torch.zeros((n, self.num_actions), dtype=f, device=dev)
        self.last_actions = torch.zeros_like(self.actions)
        self.target_dof_pos = torch.zeros_like(self.actions)

        self.dof_pos = torch.zeros((n, self.num_actions), dtype=f, device=dev)
        self.dof_vel = torch.zeros((n, self.num_actions), dtype=f, device=dev)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        self.base_pos = torch.zeros((n, 3), dtype=f, device=dev)
        self.base_quat = torch.zeros((n, 4), dtype=f, device=dev)
        self.base_lin_vel = torch.zeros((n, 3), dtype=f, device=dev)
        self.base_ang_vel = torch.zeros((n, 3), dtype=f, device=dev)
        self.projected_gravity = torch.zeros((n, 3), dtype=f, device=dev)

        self.velocity_commands = torch.zeros((n, 3), dtype=f, device=dev)
        self.mob_commands = torch.zeros((n, 3), dtype=f, device=dev)
        self.commands = self.velocity_commands

        self.privileged_friction = torch.full((n, 1), self.cfg.terrain_static_friction, dtype=f, device=dev)
        self.privileged_com_offset = torch.zeros((n, 3), dtype=f, device=dev)

        self.foot_lin_vel = torch.zeros((n, 4, 3), dtype=f, device=dev)

        self.obs_buf = torch.zeros((n, self.cfg.num_obs), dtype=f, device=dev)
        self.critic_obs_buf = torch.zeros((n, self.cfg.num_critic_obs), dtype=f, device=dev)

        self.rew_buf = torch.zeros((n,), dtype=f, device=dev)
        self.reset_buf = torch.ones((n,), dtype=gs.tc_bool, device=dev)
        self.episode_length_buf = torch.zeros((n,), dtype=gs.tc_int, device=dev)
        self.origin_xy = torch.zeros((n, 2), dtype=f, device=dev)

        self.extras: Dict[str, Any] = {
            "train_version": self.cfg.train_version,
            "obs_components": policy_obs_component_names(self.cfg),
            "critic_obs_components": critic_obs_component_names(self.cfg),
            "num_obs": self.cfg.num_obs,
            "num_critic_obs": self.cfg.num_critic_obs,
            "action_spec": self._action_spec(),
            "num_actions": self.num_actions,
            "device": str(self.device),
            "vectorized_reset": True,
            "control_dt": self.cfg.control_dt,
            "sim_dt": self.cfg.sim.sim_dt,
            "decimation": self.cfg.sim.decimation,
        }

    def _action_spec(self) -> dict:
        from envs.actions import action_spec

        return action_spec(self.cfg.robot.joint_names, self.cfg.action)

    def resample_commands(self, envs_idx: torch.Tensor | None) -> None:
        resample_commands_v1(self, envs_idx)
        sample_privileged_state(self, envs_idx)

    def _maybe_resample_commands(self) -> None:
        return

    def _build_reward_registry(self) -> None:
        pass

    def _reset_episode_reward_sums(self, env_ids: torch.Tensor | None = None) -> None:
        return

    def compute_rewards(self) -> torch.Tensor:
        return compute_v1_rewards(self)

    def get_observations(self) -> TensorDict:
        return TensorDict(
            {"policy": self.obs_buf, "critic": self.critic_obs_buf},
            batch_size=[self.num_envs],
        )

    def step(self, actions: torch.Tensor) -> Tuple[TensorDict, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if actions.device != self.device:
            actions = actions.to(device=self.device, dtype=gs.tc_float)

        from envs.actions import (
            actions_to_target_dof_pos,
            clip_policy_actions,
            select_executed_actions,
        )

        self.actions = clip_policy_actions(actions, self.cfg.action)
        exec_actions = select_executed_actions(self.actions, self.last_actions, self.cfg.action)
        self.target_dof_pos = actions_to_target_dof_pos(
            exec_actions,
            self.default_dof_pos,
            self.action_scale,
            self.dof_pos_lower,
            self.dof_pos_upper,
            self.cfg.action.soft_joint_pos_limit_factor,
        )

        dec = self.cfg.sim.decimation
        for sub in range(dec):
            self.robot.control_dofs_position(
                self.target_dof_pos[:, self.actions_dof_idx],
                self.motors_dof_idx,
            )
            self.scene.step()
            if sub == dec - 1:
                self._update_state_buffers()

        self._update_foot_state()
        self._check_termination()
        self.rew_buf = self.compute_rewards()

        if self.reset_buf.any():
            from envs.vectorized_reset import reset_envs

            reset_envs(self, self.reset_buf)

        self._compute_observations()
        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def _update_foot_state(self) -> None:
        from genesis.utils.geom import inv_quat, transform_by_quat

        foot_vel_world = self.robot.get_links_vel(links_idx_local=self.foot_links_idx_local)
        inv_q = inv_quat(self.base_quat)
        for i in range(4):
            self.foot_lin_vel[:, i] = transform_by_quat(foot_vel_world[:, i], inv_q)

    def _compute_observations(self) -> None:
        write_policy_observations(self, self.obs_buf)
        write_critic_observations(self, self.critic_obs_buf)
        clip_observations(self.obs_buf, self.cfg.obs.clip_obs)
        clip_observations(self.critic_obs_buf, self.cfg.obs.clip_obs)
