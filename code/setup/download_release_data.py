#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""下载并解压 CheckList 官方 release_data.tar.gz 到 data/raw/release_data/。

只用标准库，环境 E（Py3.7）和环境 M（Py3.10+）都能运行，Windows 也能用。

    python code/setup/download_release_data.py

来源：https://github.com/marcotcr/checklist 仓库根目录的 release_data.tar.gz（MIT 许可）。
sha256 是本仓库搭建时（2026-11-04）下载到的文件的校验值；若上游更新导致不一致，脚本会报错停下，
请在 data/README.md 里记录新的校验值和下载日期，不要静默覆盖。
"""
from __future__ import print_function

import argparse
import hashlib
import os
import sys
import tarfile

try:  # py3
    from urllib.request import urlopen
except ImportError:  # pragma: no cover
    from urllib2 import urlopen  # type: ignore

URL = "https://raw.githubusercontent.com/marcotcr/checklist/master/release_data.tar.gz"
SHA256 = "061524b310a641144d2fbccb9087634e5c8d54f32ce79885c7a1c398cadfd202"
SIZE = 19366760

# 解压后应得到的文件（相对 data/raw/），用于自检
EXPECTED_FILES = [
    "release_data/sentiment/sentiment_suite.pkl",
    "release_data/sentiment/tests_n500",
    "release_data/sentiment/predictions/bert",
    "release_data/sentiment/predictions/roberta",
    "release_data/sentiment/predictions/google",
    "release_data/sentiment/predictions/amazon",
    "release_data/sentiment/predictions/microsoft",
    "release_data/qqp/qqp_suite.pkl",
    "release_data/qqp/tests_n500",
    "release_data/qqp/predictions/bert",
    "release_data/qqp/predictions/roberta",
    "release_data/squad/squad_suite.pkl",
    "release_data/squad/squad.jsonl",
    "release_data/squad/squad.json",
    "release_data/squad/predictions/bert",
]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest):
    print("下载", url)
    resp = urlopen(url)
    total = 0
    with open(dest + ".part", "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            sys.stdout.write("\r  已下载 %.1f MB" % (total / 1e6))
            sys.stdout.flush()
    print()
    os.replace(dest + ".part", dest)


def safe_extract(tar, path):
    """拒绝绝对路径和 .. 路径，避免 tar 路径穿越。"""
    base = os.path.realpath(path)
    for m in tar.getmembers():
        target = os.path.realpath(os.path.join(path, m.name))
        if not (target == base or target.startswith(base + os.sep)):
            raise RuntimeError("tar 内含可疑路径：%s" % m.name)
        if m.issym() or m.islnk():
            raise RuntimeError("tar 内含链接文件：%s" % m.name)
    tar.extractall(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tarball", help="已手动下载的 release_data.tar.gz 路径（网络不通时用）")
    ap.add_argument("--force", action="store_true", help="即使已解压也重新解压")
    args = ap.parse_args()

    if not os.path.isdir(RAW_DIR):
        os.makedirs(RAW_DIR)
    tgz = args.tarball or os.path.join(RAW_DIR, "release_data.tar.gz")

    if not os.path.exists(tgz):
        download(URL, tgz)
    digest = sha256_of(tgz)
    size = os.path.getsize(tgz)
    print("文件:", tgz)
    print("大小: %d 字节（期望 %d）" % (size, SIZE))
    print("sha256:", digest)
    if digest != SHA256:
        sys.exit("[错误] sha256 与记录值不一致（期望 %s）。请确认来源后在 data/README.md 记录。" % SHA256)

    done = all(os.path.exists(os.path.join(RAW_DIR, p)) for p in EXPECTED_FILES)
    if done and not args.force:
        print("已解压过，跳过（加 --force 重新解压）。")
    else:
        with tarfile.open(tgz, "r:gz") as tar:
            safe_extract(tar, RAW_DIR)

    missing = [p for p in EXPECTED_FILES if not os.path.exists(os.path.join(RAW_DIR, p))]
    if missing:
        sys.exit("[错误] 缺少文件：%s" % missing)
    print("\n解压后的文件（行数为文本行数）:")
    for p in EXPECTED_FILES:
        full = os.path.join(RAW_DIR, p)
        line = "  %-48s %10d 字节" % (p, os.path.getsize(full))
        if not p.endswith((".pkl", ".json")):
            with open(full, "rb") as f:
                n = sum(1 for _ in f)
            line += "  %8d 行" % n
        print(line)
    print("\n完成。数据位于", os.path.join(RAW_DIR, "release_data"))


if __name__ == "__main__":
    main()
