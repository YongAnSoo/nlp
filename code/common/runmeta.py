# -*- coding: utf-8 -*-
"""记录「运行证据」：环境版本、git 提交、输入文件校验、耗时、参数。

每个脚本运行结束都会在输出目录写一个 run_meta.json，报告「运行证据」一栏填它的相对路径即可。
保持 Python 3.6 兼容。
"""
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

from common.paths import REPO_ROOT, rel

_PACKAGES = ["checklist", "numpy", "dill", "munch", "torch", "transformers", "tokenizers",
             "datasets", "huggingface_hub", "accelerate", "spacy", "pandas"]


def sha256_of(path, limit_bytes=None):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
            if limit_bytes and n >= limit_bytes:
                break
    return h.hexdigest()


def count_lines(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def file_info(path):
    if path is None or not os.path.exists(path):
        return None
    return {"path": rel(path), "bytes": os.path.getsize(path), "sha256": sha256_of(path)}


def package_versions():
    out = {}
    for name in _PACKAGES:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:  # 未安装就跳过
            continue
    return out


def git_info():
    def run(args):
        try:
            return subprocess.check_output(["git"] + args, cwd=REPO_ROOT,
                                           stderr=subprocess.DEVNULL).decode("utf-8").strip()
        except Exception:
            return None
    status = run(["status", "--porcelain"])
    return {"commit": run(["rev-parse", "HEAD"]), "branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(status) if status is not None else None}


def device_info():
    info = {}
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except Exception:
        pass
    return info


class RunMeta(object):
    """用法：
        meta = RunMeta("evaluate_suite", args=vars(args))
        ... 运行 ...
        meta.add(n_examples=123)
        meta.save(out_dir)
    """

    def __init__(self, script, args=None):
        self.start = time.time()
        self.data = {
            "script": script,
            "argv": sys.argv,
            "args": args or {},
            "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git": git_info(),
        }

    def add(self, **kwargs):
        self.data.update(kwargs)

    def save(self, out_dir, name="run_meta.json"):
        self.data["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        self.data["elapsed_seconds"] = round(time.time() - self.start, 2)
        self.data["packages"] = package_versions()
        self.data["device"] = device_info()
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2, default=str)
        return path
