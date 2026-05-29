"""资源路径解析（相对 taili 根目录或 genesis-world 父仓库）。"""

from __future__ import annotations

from pathlib import Path

TAILI_ROOT = Path(__file__).resolve().parent.parent
GENESIS_WORLD_ROOT = TAILI_ROOT.parent


def resolve_asset_path(path: str) -> str:
    """
    将相对路径解析为绝对路径。

    依次尝试: 原路径、taili/、genesis-world/、genesis/assets/。
    """
    p = Path(path)
    if p.is_file():
        return str(p.resolve())

    candidates = (
        TAILI_ROOT / path,
        GENESIS_WORLD_ROOT / path,
        GENESIS_WORLD_ROOT / "genesis" / "assets" / path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    return path
