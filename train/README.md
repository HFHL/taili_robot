# RL 策略版本目录

每个 RL 策略版本在 `train/<version>/` 下**独立维护**配置、环境逻辑与训练入口；共享物理封装仍在 `envs/`、`configs/paths.py`、`urdf/`。

## 已注册版本

| 版本 | 目录 | 设计文档 | 默认实验名 |
|------|------|----------|------------|
| `version1` | `train/version1/` | [desing.md](version1/desing.md) | `taili-v1-flat` |

## 目录约定（以 version1 为例）

```
train/version1/
├── desing.md          # 工程规约（观测/动作/指令/奖励/超参）
├── cfg.py             # V1EnvCfg 与 build_v1_env_cfg()
├── env.py             # GenesisEnvV1
├── observations.py    # Actor 48 维 / Critic 特权观测
├── commands.py        # 站立 20% + 速度/MoB 采样
├── rewards.py         # r_task * exp(0.02 * r_aux)
├── train_cfg.py       # PPO 超参
├── train.py           # 可直接运行的 v1 入口
└── __init__.py        # 注册到 train.VERSIONS
```

## 训练

在 `genesis-world` 根目录或 `taili/` 下：

```bash
# 推荐：统一入口 + 版本号
uv run python taili/scripts/train.py -v version1 -e taili-v1-flat -B 256 --max_iterations 10 --cpu

# 或直接进入版本目录入口
uv run python taili/train/version1/train.py -B 256 --max_iterations 10 --cpu
```

旧版基线（原 `envs/` + 加法奖励）仍可用 `-v legacy`。

## 新增 version2

1. 复制 `train/version1/` 为 `train/version2/`，按新设计改 `cfg.py` / `env.py` / 奖励等。
2. 在 `train/version2/__init__.py` 中调用 `register_version("version2", ...)`。
3. 在 `train/__init__.py` 的 `_load_version_modules()` 中 `import train.version2`。
4. 运行 `scripts/train.py -v version2`。
