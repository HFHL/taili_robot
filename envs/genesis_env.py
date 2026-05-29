"""Genesis 向量环境 —— 继承 BaseEnv + RewardMixin，所有数据流使用 GPU 张量。"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import inv_quat, transform_by_quat

from configs.env_cfg import EnvCfg
from configs.paths import resolve_asset_path
from envs.base_env import BaseEnv
from envs.commands import resample_commands
from envs.seeding import make_env_generator, set_global_seed
from envs.vectorized_reset import normalize_env_ids, refresh_state_buffers, reset_envs
from envs.actions import (
    action_spec,
    actions_to_target_dof_pos,
    build_action_scale,
    clip_policy_actions,
    normalize_dof_limits,
    select_executed_actions,
)
from envs.observations import observation_component_names, write_observations
from envs.rewards import RewardMixin
from envs.termination import check_termination


class GenesisEnv(BaseEnv, RewardMixin):
    """
    基于 Genesis 物理引擎的并行 RL 环境。

    数据流概览（每个 control step）:
        actions [num_envs, num_actions]  (策略输出，裁剪后 ∈ [-clip_actions, clip_actions])
            -> target_dof_pos = q_default + action * action_scale（软限位）
            -> control_dofs_position (PD)
            -> scene.step() 物理步进
            -> 读取状态 buffer
            -> 计算观测 [num_envs, num_obs]
            -> 计算奖励 [num_envs]
            -> 检查终止 & 重置子环境
    """

    def __init__(self, cfg: EnvCfg) -> None:
        self.cfg = cfg
        self.num_envs = cfg.sim.num_envs
        self.num_actions = cfg.num_actions
        self.num_dofs = self.num_actions  # 受控 DOF 数，与 num_actions 相同
        self.max_episode_length = cfg.max_episode_length
        self.dt = cfg.sim.sim_dt

        # Genesis 须在 train.py 中 gs.init(seed=...) ；张量均在 gs.device（Mac 上多为 MPS）
        self.device = gs.device
        if self.cfg.sim.seed is not None:
            set_global_seed(self.cfg.sim.seed)
        self.command_generator = make_env_generator(
            self.cfg.sim.seed if self.cfg.sim.seed is not None else 0,
            env_index=0,
        )

        self._build_scene()
        self._setup_robot()
        self._init_buffers()
        self._build_reward_registry()

        # 首次 reset，填充观测 buffer
        self.reset()

    # ------------------------------------------------------------------
    # 场景与机器人
    # ------------------------------------------------------------------

    def _build_scene(self) -> None:
        """创建 Genesis Scene 并添加地形。"""
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.cfg.sim.sim_dt,
                substeps=self.cfg.sim.substeps,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                # TODO: 根据机器人碰撞对数量调整
                max_collision_pairs=20,
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1.0 / self.cfg.sim.sim_dt),
            ),
            show_viewer=self.cfg.show_viewer,
        )

        # 添加固定地面
        self.scene.add_entity(
            gs.morphs.URDF(
                file=resolve_asset_path(self.cfg.terrain_urdf_path),
                fixed=True,
            )
        )

    def _setup_robot(self) -> None:
        """加载 URDF 机器人并完成 scene.build。"""
        robot_cfg = self.cfg.robot

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=resolve_asset_path(robot_cfg.urdf_path),
                pos=robot_cfg.base_init_pos,
                quat=robot_cfg.base_init_quat,
            ),
        )

        # 构建并行环境；之后所有 entity API 返回带 batch 维的张量
        self.scene.build(
            n_envs=self.num_envs,
            env_spacing=self.cfg.sim.env_spacing,
        )

        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in robot_cfg.joint_names],
            dtype=gs.tc_int,
            device=gs.device,
        )
        # 将 action 列顺序映射到 Genesis 内部 DOF 顺序
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        # PD 增益
        self.robot.set_dofs_kp(
            [robot_cfg.kp] * self.num_actions,
            self.motors_dof_idx,
        )
        self.robot.set_dofs_kv(
            [robot_cfg.kd] * self.num_actions,
            self.motors_dof_idx,
        )

        # 初始关节角 [num_dofs]
        self.default_dof_pos = torch.tensor(
            [robot_cfg.default_joint_angles[name] for name in robot_cfg.joint_names],
            dtype=gs.tc_float,
            device=gs.device,
        )

        # 完整初始 qpos: base_pos(3) + base_quat(4) + dof_pos(num_dofs)
        self.init_base_pos = torch.tensor(
            robot_cfg.base_init_pos, dtype=gs.tc_float, device=gs.device
        )
        self.init_base_quat = torch.tensor(
            robot_cfg.base_init_quat, dtype=gs.tc_float, device=gs.device
        )
        self.init_dof_pos = self.default_dof_pos.clone()
        # 并行实例的初始 qpos（含各 env 的世界系位姿偏移）
        self.init_qpos = self.robot.get_qpos().clone()
        if self.init_qpos.dim() == 1:
            self.init_qpos = self.init_qpos.unsqueeze(0).expand(self.num_envs, -1).contiguous()

        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)

        self._setup_action_limits()

    def _setup_action_limits(self) -> None:
        """缓存受控关节 URDF 位置限位，供动作映射软裁剪。"""
        lower, upper = self.robot.get_dofs_limit(self.motors_dof_idx)
        self.dof_pos_lower, self.dof_pos_upper = normalize_dof_limits(
            lower, upper, self.num_actions
        )
        self.action_scale = build_action_scale(
            self.cfg.action,
            self.cfg.robot.joint_names,
            device=gs.device,
            dtype=gs.tc_float,
        )

    # ------------------------------------------------------------------
    # Buffer 初始化
    # ------------------------------------------------------------------

    def _init_buffers(self) -> None:
        """预分配所有 GPU 张量 buffer，避免 step 中频繁 malloc。"""
        n = self.num_envs
        dev = gs.device
        f = gs.tc_float

        # --- 动作 [num_envs, num_actions]，策略 ∈ [-clip_actions, clip_actions] ---
        self.actions = torch.zeros((n, self.num_actions), dtype=f, device=dev)
        self.last_actions = torch.zeros_like(self.actions)
        self.target_dof_pos = torch.zeros_like(self.actions)

        # --- 关节状态 [num_envs, num_dofs] ---
        self.dof_pos = torch.zeros((n, self.num_actions), dtype=f, device=dev)
        self.dof_vel = torch.zeros((n, self.num_actions), dtype=f, device=dev)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        # --- 基座状态 ---
        self.base_pos = torch.zeros((n, 3), dtype=f, device=dev)       # [num_envs, 3]
        self.base_quat = torch.zeros((n, 4), dtype=f, device=dev)     # [num_envs, 4]
        self.base_lin_vel = torch.zeros((n, 3), dtype=f, device=dev)
        self.base_ang_vel = torch.zeros((n, 3), dtype=f, device=dev)
        self.projected_gravity = torch.zeros((n, 3), dtype=f, device=dev)

        n_cmd = self.cfg.command.num_commands
        self.commands = torch.zeros((n, n_cmd), dtype=f, device=dev)
        self.commands_scale = torch.tensor(
            [self.cfg.obs.obs_scales["lin_vel"]] * 2
            + [self.cfg.obs.obs_scales["ang_vel"]],
            dtype=f,
            device=dev,
        )
        self.origin_xy = torch.zeros((n, 2), dtype=f, device=dev)

        num_obs = self.cfg.num_obs
        self.obs_buf = torch.zeros((n, num_obs), dtype=f, device=dev)

        # --- RL 标准 buffer ---
        self.rew_buf = torch.zeros((n,), dtype=f, device=dev)         # [num_envs]
        self.reset_buf = torch.ones((n,), dtype=gs.tc_bool, device=dev)  # [num_envs]
        self.episode_length_buf = torch.zeros((n,), dtype=gs.tc_int, device=dev)

        self.extras: Dict[str, Any] = {
            "obs_components": observation_component_names(
                self.cfg.obs, self.num_actions, self.cfg.command.num_commands
            ),
            "num_obs": num_obs,
            "action_spec": action_spec(self.cfg.robot.joint_names, self.cfg.action),
            "num_actions": self.num_actions,
            "device": str(self.device),
            "vectorized_reset": True,
        }

    # ------------------------------------------------------------------
    # 核心 RL 接口
    # ------------------------------------------------------------------

    def reset(self) -> TensorDict:
        """重置全部并行环境并返回初始观测 TensorDict。"""
        reset_envs(self, env_ids=None)
        self.reset_buf.zero_()
        self._compute_observations()
        return self.get_observations()

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """仅重置指定环境索引（供调试或自定义 Runner 使用）。"""
        reset_envs(self, normalize_env_ids(env_ids, self.num_envs, self.device))

    def step(self, actions: torch.Tensor) -> Tuple[TensorDict, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        执行一步控制循环。

        Parameters
        ----------
        actions : torch.Tensor
            形状 [num_envs, num_actions]，须位于 ``self.device``（与 ``gs.device`` 一致）。

        Returns
        -------
        dones
            本步是否终止；``True`` 的 env 已在返回前完成部分重置，下一步继续仿真。
        """
        if actions.device != self.device:
            actions = actions.to(device=self.device, dtype=gs.tc_float)
        # ========== 1. 动作空间 -> 目标关节角 -> PD ==========
        self.actions = clip_policy_actions(actions, self.cfg.action)
        exec_actions = select_executed_actions(
            self.actions, self.last_actions, self.cfg.action
        )
        self.target_dof_pos = actions_to_target_dof_pos(
            exec_actions,
            self.default_dof_pos,
            self.action_scale,
            self.dof_pos_lower,
            self.dof_pos_upper,
            self.cfg.action.soft_joint_pos_limit_factor,
        )
        self.robot.control_dofs_position(
            self.target_dof_pos[:, self.actions_dof_idx],
            self.motors_dof_idx,
        )

        # ========== 2. 步进物理引擎 ==========
        self.scene.step()

        # ========== 3. 更新状态 buffer ==========
        self._update_state_buffers()

        # ========== 4. 重采样速度指令 ==========
        self._maybe_resample_commands()

        # ========== 5. 终止条件（须在奖励之前，alive 依赖 reset_buf） ==========
        self._check_termination()

        # ========== 6. 计算奖励 ==========
        self.rew_buf = self.compute_rewards()

        # ========== 7. 张量化部分重置（仅 done 的 env；bool 掩码，与 go2_env 一致） ==========
        if self.reset_buf.any():
            reset_envs(self, self.reset_buf)

        # ========== 8. 计算观测（reset 后已刷新状态） ==========
        self._compute_observations()

        # 更新历史 buffer（供下一步奖励 / 观测使用）
        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def get_observations(self) -> TensorDict:
        """
        返回 rsl_rl 兼容的 TensorDict 格式观测。

        Returns
        -------
        TensorDict
            key="policy", shape [num_envs, num_obs]。
        """
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _update_state_buffers(self) -> None:
        """从 Genesis entity 读取最新状态，写入 GPU buffer。"""
        self.episode_length_buf += 1

        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)

    def _compute_observations(self) -> None:
        """拼接观测向量 -> obs_buf [num_envs, num_obs]。"""
        write_observations(self, self.obs_buf)

    def _maybe_resample_commands(self) -> None:
        """按间隔重采样速度指令。"""
        interval = max(1, int(self.cfg.command.resampling_time_s / self.dt))
        envs_idx = (self.episode_length_buf % interval == 0).nonzero(as_tuple=False).flatten()
        if envs_idx.numel() > 0:
            resample_commands(self, envs_idx)

    def _check_termination(self) -> None:
        """检查 episode 终止条件，结果写入 reset_buf 与 extras['time_outs']。"""
        self.reset_buf, self.extras["time_outs"] = check_termination(self)

