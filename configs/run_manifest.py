"""训练运行目录与配置说明 —— 每次基准训练生成完整 manifest 并归档前后日志。"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from configs.paths import TAILI_ROOT, resolve_asset_path

# ---------------------------------------------------------------------------
# 文档化常量（写入 config.txt，便于对照代码）
# ---------------------------------------------------------------------------

PIPELINE_STEPS = [
    "1. 动作裁剪 clip_policy_actions(a, ActionCfg)",
    "2. 可选动作延迟 select_executed_actions（simulate_action_latency）",
    "3. 映射目标关节角 actions_to_target_dof_pos → PD 位置控制",
    "4. scene.step() 物理步进（sim_dt × substeps）",
    "5. 刷新状态 buffer（基座、关节、接触等）",
    "6. 按 resampling_time_s 重采样速度指令 commands",
    "7. 终止判断 check_termination → reset_buf / time_outs",
    "8. 计算奖励 compute_rewards（权重 × sim_dt）",
    "9. 张量化部分重置 reset_envs（仅 done 的 env）",
    "10. 计算观测 _compute_observations → obs_buf",
]

REWARD_DESCRIPTIONS: dict[str, str] = {
    "alive": "未终止环境每步 +1；权重 × sim_dt",
    "tracking_lin_vel": "exp(-||cmd_xy - base_lin_vel_xy||² / tracking_sigma)",
    "tracking_ang_vel": "exp(-(cmd_yaw - base_ang_vel_z)² / tracking_sigma)",
}

TERMINATION_DESCRIPTIONS: dict[str, str] = {
    "time_out": "episode_length_buf >= max_episode_length",
    "min_base_height": "base_pos.z < min_base_height",
    "max_roll_deg": "|roll| > max_roll_deg",
    "max_pitch_deg": "|pitch| > max_pitch_deg",
    "max_xy_distance": "水平距离原点 > max_xy_distance",
    "terminate_on_base_contact": "base_link 与地面接触",
    "sim_error": "Genesis 刚体求解异常",
}


def make_run_dir(exp_name: str, logs_root: Path | None = None) -> tuple[Path, str]:
    """
    创建带时间戳的运行目录。

    Returns
    -------
    run_dir
        logs/<exp_name>/<YYYYMMDD_HHMMSS>/
    run_id
        时间戳字符串，用作 run 唯一标识
    """
    root = logs_root or (TAILI_ROOT / "logs")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / exp_name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_id


def resolve_run_dir(
    exp_name: str,
    *,
    run_id: str | None = None,
    run_dir: Path | str | None = None,
    prefer_completed: bool = True,
) -> Path:
    """
    解析训练 run 目录。

    优先级: 显式 ``run_dir`` > ``run_id`` > 按时间戳最新（可选 prefer_completed）。
    """
    if run_dir is not None:
        path = Path(run_dir).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"run 目录不存在: {path}")
        return path

    exp_root = TAILI_ROOT / "logs" / exp_name
    if not exp_root.is_dir():
        raise FileNotFoundError(f"实验目录不存在: {exp_root}")

    if run_id is not None:
        path = exp_root / run_id
        if not path.is_dir():
            raise FileNotFoundError(f"run 不存在: {path}")
        return path

    runs = sorted(
        (p for p in exp_root.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"未找到任何 run: {exp_root}")

    if prefer_completed:
        for path in runs:
            summary = path / "post_train.json"
            if summary.is_file():
                try:
                    data = json.loads(summary.read_text(encoding="utf-8"))
                    if data.get("status") == "completed":
                        return path
                except (json.JSONDecodeError, OSError):
                    continue

    return runs[0]


def find_checkpoint(run_dir: Path, ckpt: int | None = None) -> Path:
    """返回 checkpoint 路径；``ckpt is None`` 时取最大 iteration。"""
    if ckpt is not None:
        path = run_dir / f"model_{ckpt}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint 不存在: {path}")
        return path

    candidates: list[tuple[int, Path]] = []
    for path in run_dir.glob("model_*.pt"):
        stem = path.stem.removeprefix("model_")
        if stem.isdigit():
            candidates.append((int(stem), path))
    if not candidates:
        raise FileNotFoundError(f"run 目录下无 model_*.pt: {run_dir}")
    return max(candidates, key=lambda x: x[0])[1]


def _git_info() -> dict[str, str | bool]:
    """尽力收集 git 信息；非 git 仓库时返回空字段。"""
    info: dict[str, str | bool] = {
        "available": False,
        "commit": "",
        "branch": "",
        "dirty": False,
    }
    try:
        repo = TAILI_ROOT
        while repo != repo.parent and not (repo / ".git").exists():
            repo = repo.parent
        if not (repo / ".git").exists():
            return info

        def _run(*args: str) -> str:
            return subprocess.check_output(
                args, cwd=repo, stderr=subprocess.DEVNULL, text=True
            ).strip()

        info["available"] = True
        info["commit"] = _run("git", "rev-parse", "HEAD")
        info["branch"] = _run("git", "rev-parse", "--abbrev-ref", "HEAD")
        info["dirty"] = bool(_run("git", "status", "--porcelain"))
        info["repo_root"] = str(repo)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    return info


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _observation_dim(env_cfg: Any) -> int:
    """观测维度；版本化配置优先使用 env_cfg.num_obs。"""
    if hasattr(env_cfg, "num_obs"):
        return int(env_cfg.num_obs)
    obs = env_cfg.obs
    n = env_cfg.num_actions
    nc = getattr(env_cfg.command, "num_commands", 3)
    dim = 0
    if obs.include_lin_vel:
        dim += 3
    if obs.include_ang_vel:
        dim += 3
    if obs.include_projected_gravity:
        dim += 3
    if obs.include_dof_pos:
        dim += n
    if obs.include_dof_vel:
        dim += n
    if obs.include_actions:
        dim += n
    if obs.include_commands:
        dim += nc
    if getattr(obs, "include_mob_commands", False):
        dim += getattr(env_cfg.command, "num_mob_commands", 3)
    return dim


def _max_episode_length(env_cfg: Any) -> int:
    if hasattr(env_cfg, "max_episode_length"):
        return int(env_cfg.max_episode_length)
    import math

    dt = getattr(env_cfg, "control_dt", None) or env_cfg.sim.sim_dt
    return math.ceil(env_cfg.sim.episode_length_s / dt)


def _observation_component_names(env_cfg: Any) -> list[str]:
    """根据 ObsCfg 生成观测分量说明（与 envs/observations.py 一致）。"""
    obs = env_cfg.obs
    n = env_cfg.num_actions
    nc = env_cfg.command.num_commands
    names: list[str] = []
    if obs.include_lin_vel:
        names.append("base_lin_vel(3)")
    if obs.include_ang_vel:
        names.append("base_ang_vel(3)")
    if obs.include_projected_gravity:
        names.append("projected_gravity(3)")
    if obs.include_dof_pos:
        names.append(f"dof_pos({n})")
    if obs.include_dof_vel:
        names.append(f"dof_vel({n})")
    if obs.include_actions:
        names.append(f"actions({n})")
    if obs.include_commands:
        names.append(f"commands({nc})")
    return names


def build_run_metadata(
    *,
    exp_name: str,
    run_id: str,
    run_dir: Path,
    cli_args: Mapping[str, Any],
    env_cfg: Any,
    train_cfg: dict,
    genesis_init: dict[str, Any],
) -> dict[str, Any]:
    """组装完整运行元数据（机器可读）。"""
    now = datetime.now(timezone.utc)
    resolved_urdf = resolve_asset_path(env_cfg.robot.urdf_path)
    resolved_terrain = resolve_asset_path(env_cfg.terrain_urdf_path)

    obs_components = _observation_component_names(env_cfg)

    return {
        "run_id": run_id,
        "exp_name": exp_name,
        "run_dir": str(run_dir),
        "timestamps": {
            "started_at_utc": now.isoformat(),
            "started_at_local": datetime.now().astimezone().isoformat(),
        },
        "command_line": " ".join(sys.argv),
        "cli_args": dict(cli_args),
        "git": _git_info(),
        "platform": {
            "python": sys.version,
            "system": platform.platform(),
            "machine": platform.machine(),
        },
        "pipeline_steps": PIPELINE_STEPS,
        "genesis_init": genesis_init,
        "env_cfg": _to_jsonable(env_cfg),
        "train_cfg": _to_jsonable(train_cfg),
        "resolved_paths": {
            "robot_urdf": resolved_urdf,
            "terrain_urdf": resolved_terrain,
        },
        "derived": {
            "train_version": getattr(env_cfg, "train_version", "legacy"),
            "train_version_label": getattr(env_cfg, "train_version_label", ""),
            "num_envs": env_cfg.sim.num_envs,
            "num_actions": env_cfg.num_actions,
            "num_obs": _observation_dim(env_cfg),
            "num_critic_obs": getattr(env_cfg, "num_critic_obs", _observation_dim(env_cfg)),
            "max_episode_length": _max_episode_length(env_cfg),
            "control_dt": getattr(env_cfg, "control_dt", env_cfg.sim.sim_dt),
            "obs_components": obs_components,
            "reward_terms": list(env_cfg.reward.reward_weights.keys()) or ["v1_multiplicative"],
            "seed": env_cfg.sim.seed,
        },
    }


def _section(title: str, lines: list[str]) -> str:
    bar = "=" * 72
    body = "\n".join(lines)
    return f"\n{bar}\n{title}\n{bar}\n{body}\n"


def format_config_text(metadata: dict[str, Any]) -> str:
    """生成人类可读的配置说明（config.txt）。"""
    env = metadata["env_cfg"]
    train = metadata["train_cfg"]
    ts = metadata["timestamps"]
    derived = metadata["derived"]
    paths = metadata["resolved_paths"]
    git = metadata["git"]

    lines: list[str] = []

    lines.append(_section("运行信息", [
        f"实验名称 (exp_name):     {metadata['exp_name']}",
        f"运行 ID (run_id):        {metadata['run_id']}",
        f"运行目录:                {metadata['run_dir']}",
        f"开始时间 (UTC):          {ts['started_at_utc']}",
        f"开始时间 (本地):         {ts['started_at_local']}",
        f"命令行:                  {metadata['command_line']}",
        f"随机种子 (seed):         {derived['seed']}",
    ]))

    git_lines = ["Git: 不可用"]
    if git.get("available"):
        dirty = "是 (有未提交改动)" if git.get("dirty") else "否"
        git_lines = [
            f"仓库根目录:              {git.get('repo_root', '')}",
            f"分支:                    {git.get('branch', '')}",
            f"Commit:                  {git.get('commit', '')}",
            f"工作区脏:                {dirty}",
        ]
    lines.append(_section("版本控制", git_lines))

    cli = metadata["cli_args"]
    lines.append(_section("CLI 参数", [
        f"num_envs (-B):           {cli.get('num_envs')}",
        f"max_iterations:          {cli.get('max_iterations')}",
        f"seed:                    {cli.get('seed')}",
        f"show_viewer:             {cli.get('show_viewer')}",
        f"cpu:                     {cli.get('cpu')}",
    ]))

    gi = metadata["genesis_init"]
    lines.append(_section("Genesis 初始化", [
        f"backend:                 {gi.get('backend')}",
        f"precision:               {gi.get('precision')}",
        f"logging_level:           {gi.get('logging_level')}",
        f"seed:                    {gi.get('seed')}",
        f"performance_mode:        {gi.get('performance_mode')}",
    ]))

    lines.append(_section("环境步进流程 (GenesisEnv.step)", metadata["pipeline_steps"]))

    sim = env["sim"]
    lines.append(_section("仿真 (SimCfg)", [
        f"num_envs:                {sim['num_envs']}",
        f"sim_dt:                  {sim['sim_dt']} s",
        f"substeps:                {sim['substeps']}",
        f"episode_length_s:        {sim['episode_length_s']} s",
        f"max_episode_length:      {derived['max_episode_length']} steps",
        f"env_spacing:             {sim['env_spacing']}",
        f"seed:                    {sim['seed']}",
    ]))

    robot = env["robot"]
    lines.append(_section("机器人 / URDF (RobotCfg)", [
        f"urdf_path (配置):        {robot['urdf_path']}",
        f"urdf_path (解析后):      {paths['robot_urdf']}",
        f"terrain_urdf (解析后):   {paths['terrain_urdf']}",
        f"base_init_pos:           {robot['base_init_pos']}",
        f"base_init_quat:          {robot['base_init_quat']}",
        f"kp / kd:                 {robot['kp']} / {robot['kd']}",
        f"joint_names ({len(robot['joint_names'])}):",
        *[f"  - {j}" for j in robot["joint_names"]],
        "default_joint_angles:",
        *[f"  {k}: {v}" for k, v in robot["default_joint_angles"].items()],
    ]))

    action = env["action"]
    lines.append(_section("动作空间 (ActionCfg)", [
        f"control_mode:            {action['control_mode']}",
        f"clip_actions:            {action['clip_actions']}",
        f"action_scale:            {action['action_scale']} rad",
        f"action_scale_per_joint:  {action['action_scale_per_joint']}",
        f"soft_joint_pos_limit:    {action['soft_joint_pos_limit_factor']}",
        f"simulate_action_latency: {action['simulate_action_latency']}",
        "映射: q_target = q_default + clip(a) * action_scale",
    ]))

    obs = env["obs"]
    lines.append(_section("观测空间 (ObsCfg)", [
        f"num_obs:                 {derived['num_obs']}",
        f"include_lin_vel:         {obs['include_lin_vel']}",
        f"include_ang_vel:         {obs['include_ang_vel']}",
        f"include_projected_gravity: {obs['include_projected_gravity']}",
        f"include_dof_pos:         {obs['include_dof_pos']}",
        f"include_dof_vel:         {obs['include_dof_vel']}",
        f"include_actions:         {obs['include_actions']}",
        f"include_commands:        {obs['include_commands']}",
        f"obs_scales:              {obs['obs_scales']}",
        "观测分量:",
        *[f"  - {c}" for c in derived["obs_components"]],
    ]))

    cmd = env["command"]
    lines.append(_section("速度指令 (CommandCfg)", [
        f"num_commands:            {cmd['num_commands']}",
        f"lin_vel_x_range:         {cmd['lin_vel_x_range']}",
        f"lin_vel_y_range:         {cmd['lin_vel_y_range']}",
        f"ang_vel_range:           {cmd['ang_vel_range']}",
        f"resampling_time_s:       {cmd['resampling_time_s']}",
    ]))

    reward = env["reward"]
    reward_lines = [
        f"tracking_sigma:          {reward['tracking_sigma']}",
        "奖励项 (name → weight，实际乘 sim_dt):",
    ]
    for name, weight in reward["reward_weights"].items():
        desc = REWARD_DESCRIPTIONS.get(name, "见 envs/rewards.py")
        reward_lines.append(f"  - {name}: weight={weight}  |  {desc}")
    lines.append(_section("奖励函数 (RewardCfg)", reward_lines))

    term = env["termination"]
    lines.append(_section("终止条件 (TerminationCfg)", [
        f"max_roll_deg:            {term['max_roll_deg']}",
        f"max_pitch_deg:           {term['max_pitch_deg']}",
        f"min_base_height:         {term['min_base_height']}",
        f"max_xy_distance:         {term['max_xy_distance']}",
        f"terminate_on_base_contact: {term['terminate_on_base_contact']}",
        f"base_contact_link_name:  {term['base_contact_link_name']}",
        "检查项说明:",
        *[f"  - {k}: {v}" for k, v in TERMINATION_DESCRIPTIONS.items()],
    ]))

    lines.append(_section("其他环境选项", [
        f"show_viewer:             {env['show_viewer']}",
        f"terrain_urdf_path:       {env['terrain_urdf_path']}",
    ]))

    algo = train["algorithm"]
    lines.append(_section("PPO 算法 (train_cfg.algorithm)", [
        f"class_name:              {algo['class_name']}",
        f"learning_rate:           {algo['learning_rate']}",
        f"gamma:                   {algo['gamma']}",
        f"lam (GAE):               {algo['lam']}",
        f"clip_param:              {algo['clip_param']}",
        f"entropy_coef:            {algo['entropy_coef']}",
        f"desired_kl:              {algo['desired_kl']}",
        f"num_learning_epochs:     {algo['num_learning_epochs']}",
        f"num_mini_batches:        {algo['num_mini_batches']}",
        f"max_grad_norm:           {algo['max_grad_norm']}",
        f"schedule:                {algo['schedule']}",
        f"use_clipped_value_loss:  {algo['use_clipped_value_loss']}",
        f"value_loss_coef:         {algo['value_loss_coef']}",
    ]))

    actor = train["actor"]
    critic = train["critic"]
    lines.append(_section("网络结构 (train_cfg)", [
        f"actor.hidden_dims:       {actor['hidden_dims']}",
        f"actor.activation:        {actor['activation']}",
        f"actor.distribution:      {actor['distribution_cfg']}",
        f"critic.hidden_dims:      {critic['hidden_dims']}",
        f"critic.activation:       {critic['activation']}",
        f"obs_groups:              {train['obs_groups']}",
        f"num_steps_per_env:       {train['num_steps_per_env']}",
        f"save_interval:           {train['save_interval']}",
        f"logger:                  {train['logger']}",
        f"max_iterations (CLI):    {cli.get('max_iterations')}",
    ]))

    return "\n".join(lines).strip() + "\n"


def write_run_manifest(run_dir: Path, metadata: dict[str, Any]) -> None:
    """写入 config.json、config.txt、run_meta.json。"""
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    with open(run_dir / "config.txt", "w", encoding="utf-8") as f:
        f.write(format_config_text(metadata))

    meta_summary = {
        "run_id": metadata["run_id"],
        "exp_name": metadata["exp_name"],
        "run_dir": metadata["run_dir"],
        "timestamps": metadata["timestamps"],
        "seed": metadata["derived"]["seed"],
        "num_envs": metadata["derived"]["num_envs"],
        "git_commit": metadata["git"].get("commit", ""),
        "command_line": metadata["command_line"],
    }
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_summary, f, indent=2, ensure_ascii=False)


def write_pre_train_log(run_dir: Path, env: Any, *, extra: dict[str, Any] | None = None) -> None:
    """训练前环境自检日志。"""
    lines = [
        "=== 训练前环境检查 ===",
        f"时间 (本地): {datetime.now().astimezone().isoformat()}",
        "",
        f"num_envs:      {env.num_envs}",
        f"num_actions:   {env.num_actions}",
        f"num_obs:       {env.cfg.num_obs}",
        f"obs shape:     {tuple(env.obs_buf.shape)}",
        f"device:        {env.device}",
        "",
        "obs_components:",
        *[f"  - {c}" for c in env.extras.get("obs_components", [])],
        "",
        "action_spec:",
    ]
    action_spec = env.extras.get("action_spec", {})
    for k, v in action_spec.items():
        lines.append(f"  {k}: {v}")

    if extra:
        lines.extend(["", "extra:"])
        for k, v in extra.items():
            lines.append(f"  {k}: {v}")

    lines.append("")
    with open(run_dir / "pre_train.log", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_post_train_summary(
    run_dir: Path,
    *,
    status: str,
    started_at_utc: str,
    max_iterations: int,
    iterations_completed: int | None = None,
    error: str | None = None,
) -> None:
    """训练结束后写入结果摘要。"""
    ended = datetime.now(timezone.utc)
    try:
        started = datetime.fromisoformat(started_at_utc)
        duration_s = (ended - started).total_seconds()
    except ValueError:
        duration_s = None

    checkpoints = sorted(run_dir.glob("model_*.pt"))
    tb_events = sorted(run_dir.glob("events.out.tfevents.*"))

    summary = {
        "status": status,
        "timestamps": {
            "started_at_utc": started_at_utc,
            "ended_at_utc": ended.isoformat(),
            "ended_at_local": datetime.now().astimezone().isoformat(),
            "duration_s": duration_s,
        },
        "training": {
            "max_iterations_requested": max_iterations,
            "iterations_completed": iterations_completed,
        },
        "artifacts": {
            "checkpoints": [p.name for p in checkpoints],
            "tensorboard_events": [p.name for p in tb_events],
            "config_files": [
                "config.json",
                "config.txt",
                "run_meta.json",
                "cfgs.pkl",
                "pre_train.log",
                "train.log",
            ],
        },
    }
    if error:
        summary["error"] = error

    with open(run_dir / "post_train.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    checkpoint_lines = [f"  - {p.name}" for p in checkpoints] or ["  (无)"]
    tb_lines = [f"  - {p.name}" for p in tb_events] or ["  (无)"]

    lines = [
        "=== 训练结果摘要 ===",
        f"状态:              {status}",
        f"结束时间 (UTC):    {ended.isoformat()}",
        f"耗时 (秒):         {duration_s}",
        f"请求迭代数:        {max_iterations}",
        f"完成迭代数:        {iterations_completed if iterations_completed is not None else 'N/A'}",
        "",
        f"检查点 ({len(checkpoints)}):",
        *checkpoint_lines,
        "",
        f"TensorBoard 事件 ({len(tb_events)}):",
        *tb_lines,
    ]
    if error:
        lines.extend(["", f"错误: {error}"])

    with open(run_dir / "post_train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


class _TeeStream:
    """将 stdout/stderr 同时写入终端与文件。"""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> None:
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self) -> None:
        for s in self._streams:
            s.flush()

    def fileno(self) -> int:
        return self._streams[0].fileno()

    def isatty(self) -> bool:
        return hasattr(self._streams[0], "isatty") and self._streams[0].isatty()


@contextmanager
def tee_train_log(run_dir: Path):
    """上下文管理器：训练过程控制台输出追加写入 train.log。"""
    log_path = run_dir / "train.log"
    log_file = open(log_path, "a", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(old_out, log_file)
    sys.stderr = _TeeStream(old_err, log_file)
    try:
        yield log_path
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        log_file.close()
