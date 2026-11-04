# -*- coding: utf-8 -*-
"""仓库内统一路径。保持 Python 3.6 兼容（环境 E 也会 import）。"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CODE_DIR = os.path.join(REPO_ROOT, "code")
CONFIG_DIR = os.path.join(REPO_ROOT, "configs")
DATA_DIR = os.path.join(REPO_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RELEASE_DIR = os.path.join(RAW_DIR, "release_data")
GENERATED_DIR = os.path.join(DATA_DIR, "generated")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")


def add_code_dir_to_path():
    """让 `python code/xxx/yyy.py` 直接运行时也能 `from common import ...`。"""
    if CODE_DIR not in sys.path:
        sys.path.insert(0, CODE_DIR)


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)
    return path


def rel(path):
    """返回相对仓库根目录的路径（写进报告 / run_meta.json 用）。"""
    try:
        return os.path.relpath(os.path.abspath(path), REPO_ROOT).replace(os.sep, "/")
    except ValueError:  # Windows 跨盘符
        return path
