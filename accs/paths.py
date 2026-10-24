"""路径解析：源码运行 vs PyInstaller 打包运行。

- 源码：项目根 = accs/ 的上一级
- 打包：可执行文件所在目录 = 项目根（models/、config.yaml、output/ 都放它旁边）
- 静态资源：打包后由 PyInstaller 解压到 sys._MEIPASS/accs/static
"""
from __future__ import annotations

import os
import sys


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_root() -> str:
    """项目根目录（models/、config.yaml 的父目录）。"""
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def static_dir() -> str:
    if is_frozen():
        base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
        return os.path.join(base, "accs", "static")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def models_dir() -> str:
    return os.path.join(app_root(), "models")
