
# URDF 机器人强化学习完整训练标准流程 (SOP)

## 阶段一：物理模型验证与引擎初始化 (Sanity Check)

*不要急于上神经网络，先确保物理世界是符合常理的。*

1. **URDF 静态检查：**
* 运行 `python scripts/verify_urdf.py`（引入 `urdf/taili_quad` 包，默认检查 `taili_quad.urdf`）。
* 检查质量（mass）和惯性张量（inertia）是否合理，避免出现 0 或者过大的异常值。
* 检查关节限制（joint limits：力矩、速度、位置范围）是否与真实硬件一致。


2. **Genesis 场景可视化加载：**
* 运行 `python scripts/load_viewer.py`（默认 `show_viewer=True`，加载 `urdf/taili_quad`）。
* 被动下落测试: `python scripts/load_viewer.py --passive`。
* 观察模型是否悬空、穿模或瞬间爆炸（这通常是碰撞体设置不当导致的）。


3. **被动/随机动作测试：**
* 运行 `python scripts/passive_random_test.py`（默认先零力矩 500 步，再随机位置 500 步）。
* 仅被动下落: `--mode passive`；仅随机关节: `--mode random`。
* **目标：** 确认机器人能在重力下自然下落，关节运动符合预期，没有奇异的物理抖动。



## 阶段二：环境封装与强化学习接口对接 (Environment Wrapping)

*将物理世界转化为神经网络可以理解的“张量（Tensors）”。*

1. **定义观测空间 (Observation Space)：**（已实现）
* 实现位置：`envs/genesis_env.py` + `envs/observations.py`，配置见 `configs/env_cfg.py` 的 `ObsCfg`。
* 默认 45 维（`num_actions=12`）：机体系线速度(3) + 角速度(3) + 重力投影(3) + 关节角偏差(12) + 关节速度(12) + 上一步动作(12)。
* 线/角速度在机体系下由 `transform_by_quat` 得到；重力投影为世界重力 `[0,0,-1]` 在机体系的表示。
* **输出：** `obs_buf` 形状 `[num_envs, num_obs]`，`get_observations()` 返回 `TensorDict({"policy": obs_buf})`。


2. **定义动作空间 (Action Space) 与控制映射：**（已实现）
* 实现位置：`envs/actions.py` + `configs/env_cfg.py` 的 `ActionCfg`，在 `GenesisEnv.step()` 中调用。
* 形状 `[num_envs, 12]`，列顺序同 `joint_names`（FR/FL/RR/RL × hip/thigh/calf）。
* 映射：`q_target = q_default + clip(a, ±1) * action_scale`（默认 scale=0.25 rad），再软限位裁剪后送 Genesis 位置 PD。
* 可选：`simulate_action_latency=True` 执行上一帧动作；`action_scale_per_joint` 逐关节缩放。


3. **奖励函数与终止条件：**（已实现）
* 终止：`envs/termination.py` — 超时、基座过低、roll/pitch 过大、`base_link` 触地、水平距离 > `max_xy_distance`、仿真错误。
* 奖励：`envs/rewards.py` — 默认 `alive`(+1)、`tracking_lin_vel`、`tracking_ang_vel`（高斯跟踪，权重见 `RewardCfg`）。
* 指令：`envs/commands.py` + `CommandCfg`，reset 与每 `resampling_time_s` 重采样；观测默认包含 commands(3)。


4. **随机数种子与张量化重置：**（已实现）
* `envs/vectorized_reset.py`：`step()` 内仅对 `reset_buf` 为 True 的 env 调用 `reset_envs`；reset 后 `refresh_state_buffers` 避免观测用旧状态。
* `envs/seeding.py` + `SimCfg.seed`：`gs.init(seed=...)` 与 `set_global_seed`；Mac 上 `gs.gpu` 即 MPS，张量仍在 `gs.device`。
* 自检：`uv run python taili/scripts/check_vectorized_reset.py`（可加 `--cpu`）。



## 阶段三：首次基线训练 (Baseline Training)

*跑通整个 Pipeline，验证代码没有逻辑死结。*

1. **连接 RL 算法库：**
* 将封装好的环境实例传给 PPO 算法框架（如 `rsl_rl` 或 `CleanRL`）。


2. **配置极简超参数：**
* 在 `train_cfg.py` 中设置标准的 PPO 超参数（如学习率 `1e-3`，折扣因子 `gamma=0.99`，GAE `tau=0.95`）。


3. **启动无头模式训练 (Headless Training)：**
* 关闭 Genesis 的 Viewer 以获得最大 FPS。
* 运行 `uv run python scripts/train.py -e <exp_name> -B <num_envs> --max_iterations <N> --seed <seed>`。
* **每次 run 自动归档：** 训练前在 `logs/<exp_name>/<YYYYMMDD_HHMMSS>/` 写入 `config.txt`（完整配置说明：步进流程、URDF、奖励/终止、PPO 超参、种子等）、`config.json`、`pre_train.log`；训练中镜像 `train.log` 与 TensorBoard；训练后写入 `post_train.json`。
* 监控训练日志（使用 TensorBoard 或 Wandb）。
* **核心关注指标：** * `Policy Entropy`（策略熵）是否在平稳下降。
* `Value Function Loss`（价值损失）是否在收敛。
* `Mean Reward`（平均奖励）是否在逐渐上升。





## 阶段四：分析、诊断与迭代 (The RL Loop)

*这是最耗时的一步，你会在这里不断循环，直到行为达到预期。*

1. **可视化策略 (Play/Inference)：**（已实现 `scripts/play.py`）
* 从训练 run 目录加载 `cfgs.pkl` + `model_*.pt`（与训练配置一致，避免手动改参）。
* 打开 Genesis Viewer，观察机器人实际行为。

```bash
# genesis-world 根目录 —— 默认最新 completed run + 最大 checkpoint
uv run python taili/scripts/play.py -e taili-baseline --ckpt 499

# 指定 run 目录
uv run python taili/scripts/play.py --run_dir taili/logs/taili-baseline/20260529_165616

# 无头冒烟（不打开 Viewer，跑 200 步验证加载）
uv run python taili/scripts/play.py -e taili-baseline --no-viewer --max_steps 200
```

2. **自动诊断训练指标：**（`scripts/analyze_run.py`）
* 解析 `train.log`，对比 iter 0 与末 iter 的 reward / episode length / entropy / value loss。
* 输出阶段四改进建议（终止过严、reward 平台、entropy 过高等）。

```bash
uv run python taili/scripts/analyze_run.py -e taili-baseline
tensorboard --logdir taili/logs/taili-baseline/<run_id>
```

3. **诊断问题并修改设计（核心循环）：**
* **现象：** 机器人抽搐、动作极度不自然。
* **动作：** 在 `rewards.py` 中增加对动作变化率（Action Rate）和关节扭矩的惩罚项。


* **现象：** 机器人为了快速获得奖励，发现了物理引擎的漏洞（比如靠疯狂挥动手臂向前滑行，而不是走路）。
* **动作：** 修改奖励公式，惩罚非预期的接触力，或者调整物理仿真参数。


* **现象：** 机器人学不会目标任务，奖励值一直上不去。
* **动作：** 检查观测空间是否缺少必要信息，或者尝试调整学习率、增加环境数量 (`num_envs`)。




4. **重新训练与对比：**
* 每次修改了代码或参数后，做好版本记录（git commit + 对比 `logs/<exp_name>/*/config.txt`），重新进行阶段三。
* 对比 TensorBoard 或 `analyze_run.py` 输出，保留改进最明显的 run。



## 阶段五：域随机化与 Sim2Real（进阶/可选）

*如果你最终打算把代码部署到真实的物理机器人上。*

1. **引入域随机化 (Domain Randomization)：**
* 在 `reset()` 环节，给每个环境的机器人质量、关节摩擦力、质心位置加入随机噪声（+/- 10% 到 20%）。
* 给观测数据注入高斯噪声，模拟真实传感器的误差。


2. **动作延迟模拟：**
* 在真实世界中，发送指令到电机响应有延迟。在环境的 `step()` 中缓存上一帧或上两帧的动作，模拟真实延迟，逼迫神经网络学习到更鲁棒的策略。


3. **最终导出与部署：**
* 将训练好的 PyTorch 模型导出为 ONNX 格式，准备部署到真实的下位机或树莓派/NVIDIA Jetson 等边缘计算设备上。

