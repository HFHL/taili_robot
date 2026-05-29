# taili

在 Genesis 物理引擎中对 **taili_quad** 四足机器人（12 个转动关节）做强化学习训练的项目骨架。当前代码以环境与训练入口为主，若干配置项仍为占位（见 `configs/env_cfg.py`、`scripts/train.py` 中的 TODO）。

## 任务目标

- 在仿真中加载 `urdf/taili_quad/urdf/taili_quad.urdf`，通过策略网络输出关节位置目标（PD 控制），学习稳定运动或后续指定的 locomotion 任务。
- 观测、奖励、终止条件可配置；奖励项在 `envs/rewards.py` 中实现，权重在 `configs/env_cfg.py` 的 `RewardCfg` 中定义。
- 更完整的阶段划分（URDF 检查 → 环境封装 → 基线训练 → 调参迭代 → 可选 Sim2Real）见 [`pipeline.md`](pipeline.md)。

URDF 静态校验：运行 `python scripts/verify_urdf.py`（默认加载 `urdf/taili_quad` 包内 URDF），报告写入 `urdf/taili_quad/verify_report.{txt,json}`。

## 框架选型

| 层级 | 选型 | 说明 |
|------|------|------|
| 物理仿真 | [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) | 父仓库 `genesis-world`；GPU 并行 `num_envs`，张量在 `gs.device` |
| 数值计算 | PyTorch | 观测、动作、奖励均为 batch 张量 |
| RL 算法 | PPO via `rsl-rl-lib`（≥ 5.0.0） | `scripts/train.py` 使用 `rsl_rl.runners.OnPolicyRunner`；未安装时仅实例化环境并退出 |
| 备选 | CleanRL | 代码中留有注释占位，未接入 |
| 环境接口 | 自定义 `BaseEnv` + `TensorDict` | 与 rsl_rl 的 `step` / `reset` 约定对齐 |

依赖 Genesis 与（默认）父仓库中的地面 URDF：`genesis-world/urdf/plane/plane.urdf`。本目录下无 `plane.urdf`，运行训练前需在 `EnvCfg.terrain_urdf_path` 或工作目录上处理该路径。

`urdf/taili_quad/.taili_generated/` 为 Isaac Lab / robot_lab 侧生成的配置与 PPO 超参，**不属于** 本仓库 Genesis + `GenesisEnv` 训练路径。

## 项目目录

```
taili/
├── configs/
│   ├── env_cfg.py          # 仿真、机器人、观测、奖励、终止等 dataclass 配置
│   ├── paths.py            # URDF / 资源路径解析
│   └── run_manifest.py     # 每次训练的运行目录、配置说明与前后日志
├── envs/
│   ├── base_env.py         # 向量环境抽象接口
│   ├── genesis_env.py      # Genesis 场景、步进、观测、重置
│   ├── observations.py   # 观测拼接与维度说明
│   ├── actions.py          # 动作裁剪、缩放与 PD 目标映射
│   ├── commands.py         # 速度指令采样
│   ├── termination.py      # 终止条件
│   ├── vectorized_reset.py # 部分环境重置 + 状态刷新
│   ├── seeding.py          # 随机种子
│   └── rewards.py          # 奖励 Mixin 与具体奖励项
├── scripts/
│   ├── verify_urdf.py      # URDF 静态检查（无 Genesis 依赖）
│   ├── load_viewer.py      # Genesis Viewer 加载 taili_quad
│   ├── passive_random_test.py  # 零力矩 / 随机关节动作测试
│   ├── check_vectorized_reset.py  # 验证部分 env 重置
│   ├── train.py            # 训练入口：gs.init → GenesisEnv → OnPolicyRunner
│   ├── play.py             # 阶段四：加载 checkpoint + Viewer 推理
│   └── analyze_run.py      # 阶段四：解析 train.log 指标与诊断建议
├── urdf/
│   └── taili_quad/
│       ├── urdf/taili_quad.urdf
│       ├── verify_report.{txt,json}
│       └── .taili_generated/   # Isaac Lab 生成物（参考用）
├── train/                  # RL 策略按版本分目录（见 train/README.md）
│   └── version1/           # 第一版：平地盲跑 + 站立（desing.md）
├── pipeline.md             # 训练流程 SOP
└── README.md
```

训练日志按 **实验名 + 时间戳** 写入 `logs/<exp_name>/<YYYYMMDD_HHMMSS>/`。每次运行会先写出完整配置说明，再启动训练，结束后写入结果摘要。详见下文「训练运行目录」。

## 基本训练流程

1. **环境准备**  
   在 `genesis-world` 根目录按 [AGENTS.md](../AGENTS.md) 安装依赖。

   **`uv run` 使用项目 `.venv`，不是当前 conda 环境。** 若 `uv run` 报 `No module named 'torch'`，在仓库根目录执行（先 `conda deactivate` 亦可）：

   ```bash
   cd /path/to/genesis-world
   uv sync
   uv pip install torch --python .venv/bin/python
   uv run python taili/scripts/load_viewer.py
   ```

   若坚持用 conda（例如已 `conda activate robot`），不要用 `uv run`，改为：

   ```bash
   pip install -e ".[dev]"   # 在 genesis-world 根目录，把本仓库装进 robot 环境
   python taili/scripts/load_viewer.py
   ```

   可选：`pip install rsl-rl-lib>=5.0.0`。

2. **填写配置**  
   - **Version 1（当前主路径）：** 改 `train/version1/cfg.py` 与 `desing.md` 对齐；环境逻辑在 `train/version1/env.py` 等。  
   - **Legacy 基线：** `configs/env_cfg.py` + `envs/`，训练时 `-v legacy`。  
   - 地面 URDF：将 `terrain_urdf_path` 指到可访问的 `plane.urdf`，或从仓库根目录运行并沿用父路径。

3. **可选：仅验证物理**  
   `python scripts/load_viewer.py` 在 Viewer 中加载 taili_quad；`python scripts/passive_random_test.py` 做零力矩 / 随机关节测试（见 `pipeline.md` 阶段一）。训练侧可用 `EnvCfg.show_viewer=True` 少开 `num_envs` 调试。

4. **启动训练**  
   ```bash
   cd /path/to/genesis-world
   uv run python taili/scripts/train.py -v version1 -e taili-v1-flat -B 4096 --max_iterations 1000
   ```  
   常用参数：`-v` 策略版本、`-B` 并行环境数、`--seed`、`--show_viewer`（调试时打开 Viewer）。  
   每次运行会在 `logs/<exp_name>/<时间戳>/` 下新建独立目录，**不会覆盖** 历史 run。详见 [`train/README.md`](train/README.md)。

5. **训练循环**  
   `GenesisEnv.step()`：动作裁剪 → PD 目标位置 → `scene.step()` → 终止判断 → 奖励 → 子环境重置 → 观测。  
   调奖励与终止后重新训练；指标与迭代方法见 `pipeline.md` 阶段三、四。

未安装 `rsl-rl-lib` 时，`train.py` 仍会写出配置说明与环境自检日志（`pre_train.log`），但不会执行 PPO 更新（`post_train.json` 中 `status` 为 `env_only`）。

## 训练运行目录

每次执行 `scripts/train.py` 会在 `logs/<exp_name>/<YYYYMMDD_HHMMSS>/` 创建一次独立 run，典型内容：

| 文件 | 时机 | 说明 |
|------|------|------|
| `config.txt` | 训练前 | **人类可读** 完整配置：步进流程、URDF、关节、观测/动作/奖励/终止、PPO 超参、种子、CLI |
| `config.json` | 训练前 | 同上内容的机器可读 JSON |
| `run_meta.json` | 训练前 | 运行摘要（run_id、时间戳、seed、git commit、命令行） |
| `cfgs.pkl` | 训练前 | `env_cfg` + `train_cfg` pickle，便于复现 |
| `pre_train.log` | 环境创建后 | 张量形状、设备、观测分量、有效奖励权重等自检 |
| `train.log` | 训练中 | 控制台 stdout/stderr 镜像 |
| `post_train.txt` / `post_train.json` | 训练后 | 状态、耗时、检查点与 TensorBoard 文件列表 |
| `model_*.pt` | 训练中 | rsl_rl 保存的策略检查点 |
| `events.out.tfevents.*` | 训练中 | TensorBoard 标量日志 |

查看某次 run 的配置：

```bash
cat logs/taili-locomotion/20260529_143022/config.txt
tensorboard --logdir logs/taili-locomotion/20260529_143022
```

## 阶段四：Play 与诊断

基线训练完成后：

```bash
# 1. 自动诊断指标趋势
uv run python scripts/analyze_run.py -e taili-baseline

# 2. Viewer 中目视策略（默认最新 completed run）
uv run python scripts/play.py -e taili-baseline --ckpt 499

# 3. TensorBoard 看曲线
tensorboard --logdir logs/taili-baseline/<run_id>
```

根据 `analyze_run.py` 建议或 Viewer 现象，修改 `configs/env_cfg.py` / `envs/rewards.py` 后重新 `train.py`，并用 `diff logs/<exp>/*/config.txt` 对比 run。详见 [`pipeline.md`](pipeline.md) 阶段四。
