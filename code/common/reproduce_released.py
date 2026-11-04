#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""【环境 E】用作者公开的预测（release_data/*/predictions/*）在官方 suite 上复算失败率。

用途有两个：
  1) 环境校验：作者的预测 + 作者的套件，算出来的数应与论文表 1/2/3 一致；对不上说明环境有问题
     （例如 Python 版本不对导致期望函数行为不同），这时先别跑自己的模型。
  2) 「原论文目标结果」：报告结果表第一行的原值，用这里的输出与论文 PDF 逐项核对后填写。

  python code/common/reproduce_released.py --task all

输出：results/released_reference/<task>/<model>/{per_test.csv, summary.md, per_example.jsonl.gz, run_meta.json}
      以及 results/released_reference/<task>/table.{csv,md}（各模型并排）
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.paths import RESULTS_DIR, ensure_dir, rel  # noqa: E402
from common.runmeta import RunMeta, file_info  # noqa: E402
from common import suite_io  # noqa: E402
from common import make_table  # noqa: E402


def run_task(task):
    out_root = ensure_dir(os.path.join(RESULTS_DIR, "released_reference", task))
    per_test_paths = []
    for model in suite_io.TASKS[task]["released_models"]:
        meta = RunMeta("reproduce_released", {"task": task, "model": model})
        pred = suite_io.released_pred_path(task, model)
        suite = suite_io.load_suite(task)  # 每个模型重新加载，避免结果残留
        preds, confs = suite_io.read_predictions(task, pred)
        out_dir = ensure_dir(os.path.join(out_root, model))
        label = "%s(released)" % model
        rows, n = suite_io.evaluate_to_dir(suite, task, "released", preds, confs, label, out_dir,
                                           inputs_path=suite_io.input_path(task, "released"))
        meta.add(pred_file=file_info(pred), n_predictions=len(preds), n_tests=len(rows),
                 n_testcase_records=n)
        meta.save(out_dir)
        per_test_paths.append((label, os.path.join(out_dir, "per_test.csv")))
        print("[%s] %s: %d 个测试 -> %s" % (task, model, len(rows), rel(out_dir)))
    make_table.build(per_test_paths, os.path.join(out_root, "table"),
                     title="%s：作者公开预测的复算（对照原文 %s）" % (task, suite_io.TASKS[task]["paper_table"]))
    print("  并排表:", rel(os.path.join(out_root, "table.md")))


def main():
    ap = argparse.ArgumentParser(description="复算作者公开预测")
    ap.add_argument("--task", default="all", choices=list(suite_io.TASKS) + ["all"])
    args = ap.parse_args()
    for t in (list(suite_io.TASKS) if args.task == "all" else [args.task]):
        run_task(t)


if __name__ == "__main__":
    main()
