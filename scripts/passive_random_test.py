#!/usr/bin/env python3
"""
pipeline 阶段一：被动 / 随机动作测试。

- passive：12 关节零力矩，仅重力，观察自然下落与姿态。
- random：在 URDF 限位内随机位置目标 + 弱 PD，观察关节响应是否异常。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

import genesis as gs  # noqa: E402

from load_viewer import (  # noqa: E402
    DEFAULT_BASE_POS,
    DEFAULT_BASE_QUAT,
    _standing_joint_vector,
    build_scene,
)
from urdf.taili_quad import ACTUATED_JOINT_NAMES  # noqa: E402

Mode = Literal["passive", "random"]


def _to_numpy(x: Any) -> np.ndarray:
    """Genesis 在 GPU/MPS 上返回 torch.Tensor，需先 .cpu() 再转 numpy。"""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _motors_dof_idx(robot: Any) -> list[int]:
    return [robot.get_joint(name).dofs_idx_local[0] for name in ACTUATED_JOINT_NAMES]


def _init_robot_pose(robot: Any, motors_dof_idx: list[int], *, standing: bool) -> np.ndarray:
    if standing:
        q_joint = _standing_joint_vector()
    else:
        lower, upper = robot.get_dofs_limit(motors_dof_idx)
        lower = _to_numpy(lower).reshape(-1)
        upper = _to_numpy(upper).reshape(-1)
        q_joint = lower + (upper - lower) * np.random.rand(len(motors_dof_idx))

    init_qpos = np.concatenate(
        [
            np.array(DEFAULT_BASE_POS, dtype=np.float64),
            np.array(DEFAULT_BASE_QUAT, dtype=np.float64),
            q_joint,
        ]
    )
    robot.set_qpos(init_qpos)
    robot.set_dofs_position(q_joint, motors_dof_idx)
    return q_joint


def _disable_pd(robot: Any, motors_dof_idx: list[int]) -> None:
    n = len(motors_dof_idx)
    robot.set_dofs_kp([0.0] * n, motors_dof_idx)
    robot.set_dofs_kv([0.0] * n, motors_dof_idx)


def _apply_zero_torque(robot: Any, motors_dof_idx: list[int]) -> None:
    robot.control_dofs_force(np.zeros(len(motors_dof_idx)), motors_dof_idx)


def _sample_random_targets(robot: Any, motors_dof_idx: list[int]) -> np.ndarray:
    lower, upper = robot.get_dofs_limit(motors_dof_idx)
    lower = _to_numpy(lower).reshape(-1)
    upper = _to_numpy(upper).reshape(-1)
    return lower + (upper - lower) * np.random.rand(len(motors_dof_idx))


def _read_base_height(robot: Any) -> float:
    qpos = _to_numpy(robot.get_qpos()).reshape(-1)
    return float(qpos[2])


def _read_max_joint_speed(robot: Any, motors_dof_idx: list[int]) -> float:
    vel = _to_numpy(robot.get_dofs_velocity(motors_dof_idx)).reshape(-1)
    return float(np.max(np.abs(vel)))


def run_passive(
    robot: Any,
    scene: gs.Scene,
    motors_dof_idx: list[int],
    *,
    steps: int,
    log_interval: int,
) -> None:
    _disable_pd(robot, motors_dof_idx)
    print(f"[passive] {steps} 步，零力矩 + 零 PD，初始站立姿态后释放。")

    for step in range(1, steps + 1):
        _apply_zero_torque(robot, motors_dof_idx)
        scene.step()
        if step == 1 or step % log_interval == 0 or step == steps:
            print(
                f"  step {step:4d} | base_z={_read_base_height(robot):.3f} m "
                f"| max|dof_vel|={_read_max_joint_speed(robot, motors_dof_idx):.3f} rad/s"
            )


def run_random(
    robot: Any,
    scene: gs.Scene,
    motors_dof_idx: list[int],
    *,
    steps: int,
    log_interval: int,
    resample_interval: int,
    kp: float,
    kd: float,
) -> None:
    n = len(motors_dof_idx)
    robot.set_dofs_kp([kp] * n, motors_dof_idx)
    robot.set_dofs_kv([kd] * n, motors_dof_idx)
    target = _sample_random_targets(robot, motors_dof_idx)
    robot.control_dofs_position(target, motors_dof_idx)

    print(
        f"[random] {steps} 步，每 {resample_interval} 步在关节限位内重采样目标位置 "
        f"(PD kp={kp}, kd={kd})。"
    )

    for step in range(1, steps + 1):
        if step == 1 or (step - 1) % resample_interval == 0:
            target = _sample_random_targets(robot, motors_dof_idx)
            robot.control_dofs_position(target, motors_dof_idx)
        scene.step()
        if step == 1 or step % log_interval == 0 or step == steps:
            print(
                f"  step {step:4d} | base_z={_read_base_height(robot):.3f} m "
                f"| max|dof_vel|={_read_max_joint_speed(robot, motors_dof_idx):.3f} rad/s"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="taili_quad 被动 / 随机动作测试")
    parser.add_argument(
        "--mode",
        choices=("passive", "random", "both"),
        default="both",
        help="passive=零力矩；random=随机位置；both=先 passive 再 random（默认）",
    )
    parser.add_argument("--cpu", action="store_true", help="使用 CPU 后端")
    parser.add_argument("--plane-urdf", action="store_true", help="使用 plane.urdf 地形")
    parser.add_argument("--sim-dt", type=float, default=0.02, help="仿真步长 [s]")
    parser.add_argument("--steps", type=int, default=500, help="每种模式的仿真步数")
    parser.add_argument("--log-interval", type=int, default=100, help="每隔多少步打印状态")
    parser.add_argument(
        "--resample-interval",
        type=int,
        default=50,
        help="random 模式下每隔多少步重采样目标关节角",
    )
    parser.add_argument("--kp", type=float, default=40.0, help="random 模式 PD 刚度")
    parser.add_argument("--kd", type=float, default=1.0, help="random 模式 PD 阻尼")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（random / both）")
    parser.add_argument("--no-viewer", action="store_true", help="无头模式")
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="random 模式从限位内随机初始姿态开始（默认站立）",
    )
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend, seed=args.seed)

    scene, robot = build_scene(
        use_plane_urdf=args.plane_urdf,
        sim_dt=args.sim_dt,
        show_viewer=not args.no_viewer,
    )
    scene.build(n_envs=1)

    motors_dof_idx = _motors_dof_idx(robot)
    print(f"机器人 dofs={robot.n_dofs}, 受控关节={len(motors_dof_idx)}")
    print("观察要点: 是否摔倒合理、关节是否剧烈抖动/发散、是否穿模或瞬移。")

    modes: list[Mode] = []
    if args.mode in ("passive", "both"):
        modes.append("passive")
    if args.mode in ("random", "both"):
        modes.append("random")

    for i, mode in enumerate(modes):
        if args.mode == "both" and i > 0:
            print("—" * 50)
        standing = mode == "passive" or not args.random_init
        _init_robot_pose(robot, motors_dof_idx, standing=standing)
        if mode == "passive":
            run_passive(
                robot,
                scene,
                motors_dof_idx,
                steps=args.steps,
                log_interval=args.log_interval,
            )
        else:
            run_random(
                robot,
                scene,
                motors_dof_idx,
                steps=args.steps,
                log_interval=args.log_interval,
                resample_interval=args.resample_interval,
                kp=args.kp,
                kd=args.kd,
            )

    print("测试结束。关闭 Viewer 或查看终端日志判断物理是否合理。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
