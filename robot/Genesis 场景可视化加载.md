已添加 [`scripts/load_viewer.py`](scripts/load_viewer.py)，用于 pipeline 阶段一的 Genesis 可视化加载。

uv run python taili/scripts/load_viewer.py

## 功能

- 从 `urdf.taili_quad` 加载 `taili_quad.urdf`（绝对路径，不依赖 cwd）
- 地面默认 `gs.morphs.Plane()`；可选 `--plane-urdf` 使用父仓库 `genesis/assets/urdf/plane/plane.urdf`
- 初始位姿与 Isaac Lab 生成配置一致：基座 `z=0.58`，12 关节站立角（thigh `0.8`，calf `-0.5`）
- 默认用 PD 保持站立，便于在 Viewer 里观察；`--passive` 为仅重力下落

## 运行方式

在 **genesis-world 根目录**（需已 `uv sync` 并安装 PyTorch）：

```bash
cd /Users/Admin/project/genesis-world
uv run python taili/scripts/load_viewer.py
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--passive` | 零 PD，观察重力下落 |
| `--cpu` | CPU 后端（无 GPU 时） |
| `--steps 500` | 固定步数后退出 |
| `--no-viewer` | 无头冒烟测试 |

当前环境未安装 `torch`，脚本未在本机实际跑通仿真；你本地装好依赖后，直接运行上述命令即可打开 Viewer。

## 观察要点（pipeline 阶段一）

关闭窗口或 Ctrl+C 结束前，留意：

1. 初始是否穿地或悬空过高  
2. 是否瞬间飞散（质量/惯性/碰撞体问题）  
3. 关节是否异常抖动  

`pipeline.md` 与 `README.md` 已补充对应说明。下一步若要阶段一的「被动/随机动作测试」，可以说一声，我可以再加 `scripts/passive_test.py`。