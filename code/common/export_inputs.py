#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""【环境 E】从官方 suite 导出模型输入文件，并校验与官方 release 文件逐行一致。

  released 模式：不写新文件，只校验 suite 中保存的抽样与官方 tests_n500 / squad.jsonl 逐行一致，
                 并写出每个测试的行范围 data/generated/<task>/released_index.json
  full 模式    ：展开 *全部* 用例（不抽样），写 data/generated/<task>/full_inputs.* 与 full_index.json

示例：
  python code/common/export_inputs.py --task all --mode both
"""
import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.paths import GENERATED_DIR, ensure_dir, rel  # noqa: E402
from common.runmeta import RunMeta, file_info  # noqa: E402
from common import suite_io  # noqa: E402


def export(task, mode, suite_file=None, strict_python=True):
    suite = suite_io.load_suite(task, path=suite_file, strict_python=strict_python)
    examples = suite_io.example_list(suite, task, mode)
    out_dir = ensure_dir(os.path.join(GENERATED_DIR, task))
    path = suite_io.input_path(task, mode)
    if mode == "released":
        suite_io.check_alignment(task, examples, suite_io.read_input_file(task, path), path)
        print("[%s/released] 与官方文件 %s 逐行一致，共 %d 个样本" % (task, rel(path), len(examples)))
    else:
        suite_io.write_input_file(task, examples, path)
        print("[%s/full] 写出 %s，共 %d 个样本（全部用例，未抽样）" % (task, rel(path), len(examples)))
    index = suite_io.tests_index(suite)
    index_path = os.path.join(out_dir, "%s_index.json" % mode)
    with io.open(index_path, "w", encoding="utf-8") as f:
        json.dump({"task": task, "mode": mode, "n_examples": len(examples), "inputs": rel(path),
                   "tests": index}, f, ensure_ascii=False, indent=1)
    n_cases = sum(t["n_testcases_in_file"] for t in index)
    print("  %d 个测试，%d 个用例；行范围写入 %s" % (len(index), n_cases, rel(index_path)))
    return {"task": task, "mode": mode, "n_tests": len(index), "n_testcases": n_cases,
            "n_examples": len(examples), "inputs": file_info(path), "index": rel(index_path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--task", default="all", choices=list(suite_io.TASKS) + ["all"])
    ap.add_argument("--mode", default="both", choices=list(suite_io.MODES) + ["both"])
    args = ap.parse_args()
    meta = RunMeta("export_inputs", vars(args))
    tasks = list(suite_io.TASKS) if args.task == "all" else [args.task]
    modes = list(suite_io.MODES) if args.mode == "both" else [args.mode]
    out = []
    for t in tasks:
        for m in modes:
            out.append(export(t, m))
    meta.add(outputs=out)
    print("run_meta:", rel(meta.save(ensure_dir(GENERATED_DIR), "export_inputs_run_meta.json")))


if __name__ == "__main__":
    main()
