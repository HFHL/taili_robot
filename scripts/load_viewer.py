#!/usr/bin/env python3
"""
pipeline 阶段一：Genesis 场景可视化加载 taili_quad。

在 Viewer 中检查模型是否悬空、穿模或数值爆炸；可选被动下落（零力矩）或默认站立 PD。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import genesis as gs  # noqa: E402

from urdf.taili_quad import ACTUATED_JOINT_NAMES, URDF_PATH  # noqa: E402

# 与 .taili_generated/taili_quad.py 中 init_state 一致
DEFAULT_BASE_POS = (0.0, 0.0, 0.58)
DEFAULT_BASE_QUAT = (1.0, 0.0, 0.0, 0.0)
DEFAULT_JOINT_POS = {
    "FR_hip_joint": 0.0,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -0.5,
    "FL_hip_joint": 0.0,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -0.5,
    "RR_hip_joint": 0.0,
    "RR_thigh_joint": 0.8,
    "RR_calf_joint": -0.5,
    "RL_hip_joint": 0.0,
    "RL_thigh_joint": 0.8,
    "RL_calf_joint": -0.5,
}


def _standing_joint_vector() -> np.ndarray:
    return np.array([DEFAULT_JOINT_POS[name] for name in ACTUATED_JOINT_NAMES], dtype=np.float64)


def _resolve_plane_urdf() -> Path | None:
    """父仓库 plane.urdf（若存在）；否则用 gs.morphs.Plane()。"""
    candidates = [
        PROJECT_ROOT.parent / "genesis" / "assets" / "urdf" / "plane" / "plane.urdf",
        PROJECT_ROOT / "urdf" / "plane" / "plane.urdf",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def build_scene(
    *,
    use_plane_urdf: bool,
    sim_dt: float,
    show_viewer: bool,
) -> tuple[gs.Scene, Any]:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=sim_dt, substeps=2),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=False,
            max_collision_pairs=64,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, -2.0, 1.2),
            camera_lookat=(0.0, 0.0, 0.35),
            camera_fov=40,
            max_FPS=int(1.0 / sim_dt),
        ),
        show_viewer=show_viewer,
    )

    plane_urdf = _resolve_plane_urdf()
    if use_plane_urdf and plane_urdf is not None:
        scene.add_entity(gs.morphs.URDF(file=str(plane_urdf), fixed=True))
        print(f"地形: URDF plane ({plane_urdf})")
    else:
        scene.add_entity(gs.morphs.Plane())
        print("地形: gs.morphs.Plane()")

    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(URDF_PATH.resolve()),
            pos=DEFAULT_BASE_POS,
            quat=DEFAULT_BASE_QUAT,
        ),
    )
    return scene, robot


def setup_robot(
    robot: Any,
    *,
    passive: bool,
    kp: float,
    kd: float,
) -> list[int]:
    motors_dof_idx = [robot.get_joint(name).dofs_idx_local[0] for name in ACTUATED_JOINT_NAMES]
    q_joint = _standing_joint_vector()

    # 基座 free-flyer (7) + 关节
    init_qpos = np.concatenate(
        [
            np.array(DEFAULT_BASE_POS, dtype=np.float64),
            np.array(DEFAULT_BASE_QUAT, dtype=np.float64),
            q_joint,
        ]
    )
    robot.set_qpos(init_qpos)
    robot.set_dofs_position(q_joint, motors_dof_idx)

    if not passive:
        robot.set_dofs_kp([kp] * len(motors_dof_idx), motors_dof_idx)
        robot.set_dofs_kv([kd] * len(motors_dof_idx), motors_dof_idx)
        robot.control_dofs_position(q_joint, motors_dof_idx)

    print(f"机器人: {robot.name}, links={robot.n_links}, dofs={robot.n_dofs}")
    print(f"URDF: {URDF_PATH}")
    print(f"受控关节 ({len(motors_dof_idx)}): {', '.join(ACTUATED_JOINT_NAMES)}")
    print(f"模式: {'被动（零 PD，仅重力）' if passive else f'站立 PD (kp={kp}, kd={kd})'}")
    return motors_dof_idx


def main() -> int:
    parser = argparse.ArgumentParser(description="taili_quad Genesis 可视化加载")
    parser.add_argument("--cpu", action="store_true", help="使用 CPU 后端")
    parser.add_argument("--plane-urdf", action="store_true", help="使用 plane.urdf 代替内置 Plane")
    parser.add_argument("--passive", action="store_true", help="零力矩被动下落（不做 PD 保持）")
    parser.add_argument("--sim-dt", type=float, default=0.02, help="仿真步长 [s]")
    parser.add_argument("--steps", type=int, default=0, help="仿真步数，0 表示一直运行直到关闭 Viewer")
    parser.add_argument("--kp", type=float, default=80.0, help="关节位置 PD 刚度（非被动模式）")
    parser.add_argument("--kd", type=float, default=2.0, help="关节速度 PD 阻尼（非被动模式）")
    parser.add_argument("--no-viewer", action="store_true", help="无头模式（仅冒烟测试）")
    args = parser.parse_args()

    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend)

    scene, robot = build_scene(
        use_plane_urdf=args.plane_urdf,
        sim_dt=args.sim_dt,
        show_viewer=not args.no_viewer,
    )
    scene.build(n_envs=1)

    motors_dof_idx = setup_robot(robot, passive=args.passive, kp=args.kp, kd=args.kd)
    q_hold = _standing_joint_vector()

    print("开始仿真。关闭 Viewer 窗口或 Ctrl+C 结束。")
    print("观察要点: 初始是否穿地/悬空过高、是否瞬间飞散、关节是否异常抖动。")

    step = 0
    try:
        while args.steps == 0 or step < args.steps:
            if not args.passive:
                robot.control_dofs_position(q_hold, motors_dof_idx)
            scene.step()
            step += 1
    except KeyboardInterrupt:
        print("\n已中断。")

    print(f"共仿真 {step} 步。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
