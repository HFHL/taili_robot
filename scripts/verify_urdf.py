#!/usr/bin/env python3
"""taili_quad URDF 静态合规性检查（不依赖 Genesis / GPU）。"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from urdf.taili_quad import (  # noqa: E402
    ACTUATED_JOINT_NAMES,
    FOOT_LINK_NAMES,
    PACKAGE_ROOT,
    SYMMETRY_MASS_PAIRS,
    URDF_PATH,
)

# 物理合理性阈值（pipeline.md 阶段一）
MIN_MASS_KG = 1e-6
MAX_MASS_KG = 500.0
MAX_INERTIA_DIAG = 50.0
SYMMETRY_MASS_REL_TOL = 0.05
MIN_AXIS_NORM = 1e-6


@dataclass
class CheckResult:
    name: str
    status: str  # PASS | WARN | FAIL
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def _parse_float(text: str | None, default: float | None = None) -> float | None:
    if text is None or text.strip() == "":
        return default
    return float(text)


def _inertia_matrix(link: ET.Element) -> tuple[float, ...] | None:
    inertial = link.find("inertial")
    if inertial is None:
        return None
    inertia = inertial.find("inertia")
    if inertia is None:
        return None
    keys = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
    vals = []
    for k in keys:
        v = _parse_float(inertia.get(k))
        if v is None:
            return None
        vals.append(v)
    return tuple(vals)


def _is_positive_definite(ixx: float, ixy: float, ixz: float, iyy: float, iyz: float, izz: float) -> bool:
    """3x3 对称惯性矩阵正定性：顺序主子式均为正。"""
    if ixx <= 0 or iyy <= 0 or izz <= 0:
        return False
    det2 = ixx * iyy - ixy * ixy
    if det2 <= 0:
        return False
    det3 = (
        ixx * (iyy * izz - iyz * iyz)
        - ixy * (ixy * izz - ixz * iyz)
        + ixz * (ixy * iyz - ixz * iyy)
    )
    return det3 > 0


def _resolve_mesh_path(urdf_file: Path, filename: str) -> Path:
    return (urdf_file.parent / filename).resolve()


def _joint_parent_child(joint: ET.Element) -> tuple[str | None, str | None]:
    parent = joint.find("parent")
    child = joint.find("child")
    p = parent.get("link") if parent is not None else None
    c = child.get("link") if child is not None else None
    return p, c


class UrdfChecker:
    def __init__(self, urdf_path: Path, package_root: Path):
        self.urdf_path = urdf_path.resolve()
        self.package_root = package_root.resolve()
        self.tree: ET.ElementTree | None = None
        self.root: ET.Element | None = None
        self.results: list[CheckResult] = []

    def _add(self, name: str, status: str, message: str, **details: Any) -> None:
        self.results.append(CheckResult(name=name, status=status, message=message, details=details))

    def run_all(self) -> list[CheckResult]:
        self.results.clear()
        self._check_xml_parse()
        if self.root is None:
            return self.results

        self._check_robot_element()
        links, joints = self._collect_elements()
        self._check_joint_references(links, joints)
        self._check_tree_topology(links, joints)
        self._check_single_child(joints)
        self._check_inertial(links)
        self._check_inertia_positive_definite(links)
        self._check_revolute_limits(joints)
        self._check_revolute_axis(joints)
        self._check_base_link(links)
        self._check_quadruped_joints(joints)
        self._check_foot_links(links)
        self._check_mesh_files()
        self._check_symmetry(links)
        self._check_mass_inertia_sanity(links)
        return self.results

    def _check_xml_parse(self) -> None:
        try:
            self.tree = ET.parse(self.urdf_path)
            self.root = self.tree.getroot()
            self._add("xml_parse", "PASS", "XML 解析成功")
        except ET.ParseError as exc:
            self._add("xml_parse", "FAIL", f"XML 解析失败: {exc}")

    def _check_robot_element(self) -> None:
        assert self.root is not None
        if self.root.tag != "robot":
            self._add("robot_element", "FAIL", f"根元素应为 <robot>，实际为 <{self.root.tag}>")
            return
        name = self.root.get("name", "")
        self._add("robot_element", "PASS", f'根元素 <robot name="{name}">')

    def _collect_elements(self) -> tuple[dict[str, ET.Element], list[ET.Element]]:
        assert self.root is not None
        links = {el.get("name"): el for el in self.root.findall("link") if el.get("name")}
        joints = self.root.findall("joint")
        return links, joints

    def _check_joint_references(self, links: dict[str, ET.Element], joints: list[ET.Element]) -> None:
        bad: list[str] = []
        for joint in joints:
            jname = joint.get("name", "?")
            parent, child = _joint_parent_child(joint)
            if parent not in links:
                bad.append(f"{jname}: parent '{parent}' 不存在")
            if child not in links:
                bad.append(f"{jname}: child '{child}' 不存在")
        if bad:
            self._add("joint_references", "FAIL", f"{len(bad)} 处引用错误", errors=bad)
        else:
            self._add("joint_references", "PASS", f"所有 {len(joints)} 个 joint 引用合法")

    def _check_tree_topology(self, links: dict[str, ET.Element], joints: list[ET.Element]) -> None:
        child_to_parent: dict[str, str] = {}
        parent_count: dict[str, int] = {}
        for joint in joints:
            parent, child = _joint_parent_child(joint)
            if not parent or not child:
                continue
            if child in child_to_parent:
                self._add(
                    "tree_topology",
                    "FAIL",
                    f"link '{child}' 有多个 parent joint",
                    duplicate_child=child,
                )
                return
            child_to_parent[child] = parent
            parent_count[parent] = parent_count.get(parent, 0) + 1

        children = set(child_to_parent)
        roots = [name for name in links if name not in children]
        if len(roots) != 1:
            self._add("tree_topology", "FAIL", f"根 link 数量应为 1，实际 {len(roots)}", roots=roots)
            return
        root = roots[0]
        if root != "base_link":
            self._add("tree_topology", "WARN", f"根节点为 '{root}'（期望 base_link）", root=root)
        else:
            self._add(
                "tree_topology",
                "PASS",
                f"树形拓扑合法，根节点 '{root}'，共 {len(links)} 个 link",
            )

    def _check_single_child(self, joints: list[ET.Element]) -> None:
        child_count: dict[str, int] = {}
        for joint in joints:
            _, child = _joint_parent_child(joint)
            if child:
                child_count[child] = child_count.get(child, 0) + 1
        dup = {k: v for k, v in child_count.items() if v > 1}
        if dup:
            self._add("single_child", "FAIL", "存在 link 被多个 joint 作为 child", duplicates=dup)
        else:
            self._add("single_child", "PASS", "每个 link 最多被一个 joint 作为 child")

    def _check_inertial(self, links: dict[str, ET.Element]) -> None:
        missing: list[str] = []
        invalid: list[str] = []
        for name, link in links.items():
            inertial = link.find("inertial")
            if inertial is None:
                missing.append(name)
                continue
            mass_el = inertial.find("mass")
            mass = _parse_float(mass_el.get("value") if mass_el is not None else None)
            if mass is None or mass <= 0:
                invalid.append(f"{name}: mass={mass}")
                continue
            if _inertia_matrix(link) is None:
                invalid.append(f"{name}: inertia 缺失或不全")

        if missing or invalid:
            self._add(
                "inertial",
                "FAIL",
                f"缺失 {len(missing)}，无效 {len(invalid)}",
                missing=missing,
                invalid=invalid,
            )
        else:
            self._add("inertial", "PASS", f"所有 {len(links)} 个 link 均有合法 inertial")

    def _check_inertia_positive_definite(self, links: dict[str, ET.Element]) -> None:
        bad: list[str] = []
        for name, link in links.items():
            mat = _inertia_matrix(link)
            if mat is None:
                bad.append(name)
                continue
            if not _is_positive_definite(*mat):
                bad.append(name)
        if bad:
            self._add("inertia_positive_definite", "FAIL", f"{len(bad)} 个 link 惯性非正定", links=bad)
        else:
            self._add("inertia_positive_definite", "PASS", "所有惯性矩阵均正定")

    def _check_revolute_limits(self, joints: list[ET.Element]) -> None:
        revolute = [j for j in joints if j.get("type") == "revolute"]
        bad: list[str] = []
        for joint in revolute:
            jname = joint.get("name", "?")
            limit = joint.find("limit")
            if limit is None:
                bad.append(f"{jname}: 无 limit")
                continue
            lower = _parse_float(limit.get("lower"))
            upper = _parse_float(limit.get("upper"))
            effort = _parse_float(limit.get("effort"))
            velocity = _parse_float(limit.get("velocity"))
            if lower is None or upper is None or lower >= upper:
                bad.append(f"{jname}: lower={lower} upper={upper}")
            if effort is not None and effort <= 0:
                bad.append(f"{jname}: effort={effort}")
            if velocity is not None and velocity <= 0:
                bad.append(f"{jname}: velocity={velocity}")
        if bad:
            self._add("revolute_limits", "FAIL", f"{len(bad)} 个 revolute 限位异常", issues=bad)
        else:
            self._add("revolute_limits", "PASS", f"所有 {len(revolute)} 个 revolute joint 限位合法")

    def _check_revolute_axis(self, joints: list[ET.Element]) -> None:
        revolute = [j for j in joints if j.get("type") == "revolute"]
        bad: list[str] = []
        for joint in revolute:
            jname = joint.get("name", "?")
            axis = joint.find("axis")
            if axis is None:
                bad.append(f"{jname}: 无 axis")
                continue
            xyz = axis.get("xyz", "")
            parts = xyz.split()
            if len(parts) != 3:
                bad.append(f"{jname}: axis='{xyz}'")
                continue
            try:
                vec = [float(p) for p in parts]
            except ValueError:
                bad.append(f"{jname}: axis='{xyz}'")
                continue
            norm = math.sqrt(sum(v * v for v in vec))
            if norm < MIN_AXIS_NORM:
                bad.append(f"{jname}: 轴长度过小 ({norm})")
        if bad:
            self._add("revolute_axis", "FAIL", f"{len(bad)} 个 revolute 轴非法", issues=bad)
        else:
            self._add("revolute_axis", "PASS", "所有 revolute joint 轴向量合法")

    def _check_base_link(self, links: dict[str, ET.Element]) -> None:
        if "base_link" in links:
            self._add("base_link", "PASS", "存在 base_link")
        else:
            self._add("base_link", "FAIL", "缺少 base_link")

    def _check_quadruped_joints(self, joints: list[ET.Element]) -> None:
        by_name = {j.get("name"): j for j in joints if j.get("name")}
        missing = [n for n in ACTUATED_JOINT_NAMES if n not in by_name]
        wrong_type = [
            n for n in ACTUATED_JOINT_NAMES if n in by_name and by_name[n].get("type") != "revolute"
        ]
        if missing or wrong_type:
            self._add(
                "quadruped_joints",
                "FAIL",
                f"缺失 {len(missing)}，类型错误 {len(wrong_type)}",
                missing=missing,
                wrong_type=wrong_type,
            )
        else:
            self._add(
                "quadruped_joints",
                "PASS",
                f"四条腿 {len(ACTUATED_JOINT_NAMES)} 个 actuated joint 完整且均为 revolute",
            )

    def _check_foot_links(self, links: dict[str, ET.Element]) -> None:
        missing = [n for n in FOOT_LINK_NAMES if n not in links]
        if missing:
            self._add("foot_links", "FAIL", f"缺少 foot link: {missing}")
        else:
            self._add("foot_links", "PASS", "四个 foot link 完整")

    def _check_mesh_files(self) -> None:
        assert self.tree is not None
        mesh_els = self.root.findall(".//mesh") if self.root is not None else []
        missing: list[str] = []
        checked = 0
        for mesh in mesh_els:
            fn = mesh.get("filename")
            if not fn:
                continue
            checked += 1
            path = _resolve_mesh_path(self.urdf_path, fn)
            if not path.is_file():
                missing.append(str(path))
        if missing:
            self._add("mesh_files", "FAIL", f"{len(missing)} 个 mesh 文件缺失", missing=missing)
        else:
            self._add("mesh_files", "PASS", f"所有 {checked} 个 mesh 文件路径有效")

    def _link_mass(self, links: dict[str, ET.Element], name: str) -> float | None:
        link = links.get(name)
        if link is None:
            return None
        inertial = link.find("inertial")
        if inertial is None:
            return None
        mass_el = inertial.find("mass")
        return _parse_float(mass_el.get("value") if mass_el is not None else None)

    def _check_symmetry(self, links: dict[str, ET.Element]) -> None:
        over_tol: list[dict[str, Any]] = []
        for left, right in SYMMETRY_MASS_PAIRS:
            m1 = self._link_mass(links, left)
            m2 = self._link_mass(links, right)
            if m1 is None or m2 is None:
                over_tol.append({"pair": (left, right), "error": "mass missing"})
                continue
            avg = (m1 + m2) / 2.0
            if avg <= 0:
                rel = float("inf")
            else:
                rel = abs(m1 - m2) / avg
            if rel > SYMMETRY_MASS_REL_TOL:
                over_tol.append({"pair": (left, right), "m1": m1, "m2": m2, "rel_diff": rel})
        if over_tol:
            self._add(
                "symmetry",
                "WARN",
                f"{len(over_tol)} 对对称连杆质量偏差超过 {SYMMETRY_MASS_REL_TOL:.0%}",
                pairs=over_tol,
            )
        else:
            self._add(
                "symmetry",
                "PASS",
                f"对称腿质量偏差均在 {SYMMETRY_MASS_REL_TOL:.0%} 以内",
            )

    def _check_mass_inertia_sanity(self, links: dict[str, ET.Element]) -> None:
        """pipeline.md：质量与惯性对角元是否在合理范围。"""
        mass_issues: list[str] = []
        inertia_issues: list[str] = []
        for name, link in links.items():
            inertial = link.find("inertial")
            if inertial is None:
                continue
            mass_el = inertial.find("mass")
            mass = _parse_float(mass_el.get("value") if mass_el is not None else None)
            if mass is not None and (mass < MIN_MASS_KG or mass > MAX_MASS_KG):
                mass_issues.append(f"{name}: mass={mass} kg")
            mat = _inertia_matrix(link)
            if mat is not None:
                ixx, _, _, iyy, _, izz = mat
                for label, val in (("ixx", ixx), ("iyy", iyy), ("izz", izz)):
                    if val < 0 or val > MAX_INERTIA_DIAG:
                        inertia_issues.append(f"{name}: {label}={val}")

        if mass_issues or inertia_issues:
            self._add(
                "mass_inertia_sanity",
                "WARN",
                f"质量异常 {len(mass_issues)}，惯性对角异常 {len(inertia_issues)}",
                mass_issues=mass_issues,
                inertia_issues=inertia_issues,
            )
        else:
            total_mass = sum(
                self._link_mass(links, n) or 0.0 for n in links
            )
            self._add(
                "mass_inertia_sanity",
                "PASS",
                f"质量与惯性对角元在合理范围内（连杆总质量约 {total_mass:.3f} kg）",
                total_mass_kg=round(total_mass, 4),
            )


def _summarize(results: list[CheckResult]) -> dict[str, int]:
    counts = {"total": len(results), "passed": 0, "warned": 0, "failed": 0}
    for r in results:
        if r.status == "PASS":
            counts["passed"] += 1
        elif r.status == "WARN":
            counts["warned"] += 1
        else:
            counts["failed"] += 1
    return counts


def _format_txt(urdf_rel: str, results: list[CheckResult], summary: dict[str, int]) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    lines = [
        "URDF 合规性验证报告",
        "=" * 50,
        f"文件: {urdf_rel}",
        f"包路径: {PACKAGE_ROOT}",
        f"时间: {ts}",
        "",
        f"总计: {summary['total']}  通过: {summary['passed']}  "
        f"警告: {summary['warned']}  失败: {summary['failed']}",
        "=" * 50,
        "",
    ]
    for r in results:
        lines.append(f"  [{r.status}]  {r.name}")
        lines.append(f"         {r.message}")
        lines.append("")
    if summary["failed"] > 0:
        lines.append("结论: FAIL — 存在必须修复的问题后才能用于仿真训练")
    elif summary["warned"] > 0:
        lines.append("结论: WARN — 基本可用，建议处理警告项后再训练")
    else:
        lines.append("结论: PASS — URDF 完全合规，可用于仿真训练")
    return "\n".join(lines) + "\n"


def _format_json(urdf_rel: str, results: list[CheckResult], summary: dict[str, int]) -> dict[str, Any]:
    return {
        "urdf": urdf_rel,
        "package_root": str(PACKAGE_ROOT),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": summary["total"],
            "passed": summary["passed"],
            "warned": summary["warned"],
            "failed": summary["failed"],
        },
        "checks": [
            {"name": r.name, "status": r.status, "message": r.message, "details": r.details}
            for r in results
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="taili_quad URDF 静态检查")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=URDF_PATH,
        help=f"URDF 文件路径（默认: {URDF_PATH}）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PACKAGE_ROOT,
        help="报告输出目录（默认: urdf/taili_quad）",
    )
    args = parser.parse_args()

    urdf_path = args.urdf.resolve()
    if not urdf_path.is_file():
        print(f"错误: URDF 不存在: {urdf_path}", file=sys.stderr)
        return 2

    checker = UrdfChecker(urdf_path=urdf_path, package_root=PACKAGE_ROOT)
    results = checker.run_all()
    summary = _summarize(results)

    try:
        urdf_rel = urdf_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        urdf_rel = str(urdf_path)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / "verify_report.txt"
    json_path = out_dir / "verify_report.json"

    txt_path.write_text(_format_txt(urdf_rel, results, summary), encoding="utf-8")
    json_path.write_text(
        json.dumps(_format_json(urdf_rel, results, summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(_format_txt(urdf_rel, results, summary))
    print(f"报告已写入:\n  {txt_path}\n  {json_path}")

    if summary["failed"] > 0:
        return 1
    if summary["warned"] > 0:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
