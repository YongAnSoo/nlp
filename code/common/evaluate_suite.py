#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""【环境 E】用 CheckList 套件给一个预测文件打分，输出逐测试失败率与逐用例结果。

  python code/common/evaluate_suite.py --task sentiment --mode released \
      --pred results/sentiment/bert-textattack/released/preds.txt --label bert-textattack

输出（默认 <pred 所在目录>/eval/）：
  per_test.csv           每个测试一行：capability / type / 用例数 / 失败数 / 失败率
  summary.md             同上的 Markdown 表
  per_example.jsonl.gz   每个用例一行：ID、输入行号、样本、标签、预测、置信度、期望结果、是否通过
  run_meta.json          运行证据（环境、git 提交、预测文件 sha256、耗时）

预测文件格式（每行对应输入文件的一行）：
  sentiment: "pred p_neg p_neu p_pos"，pred ∈ {0,1,2}
  qqp      : "P(duplicate)"
  squad    : 预测答案字符串
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.paths import ensure_dir, rel  # noqa: E402
from common.runmeta import RunMeta, file_info  # noqa: E402
from common import suite_io  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="CheckList 打分")
    ap.add_argument("--task", required=True, choices=list(suite_io.TASKS))
    ap.add_argument("--mode", required=True, choices=list(suite_io.MODES))
    ap.add_argument("--pred", required=True, help="预测文件")
    ap.add_argument("--label", required=True, help="模型标签，写进结果（如 bert-textattack）")
    ap.add_argument("--out", help="输出目录（默认 <pred 所在目录>/eval）")
    ap.add_argument("--inputs", help="预测时用的输入文件（默认按 --mode 取标准路径），用于逐行对齐校验")
    ap.add_argument("--no-align-check", action="store_true", help="跳过与输入文件的逐行比对（不建议）")
    ap.add_argument("--suite", help="suite pkl 路径（默认 data/raw/release_data/<task>/<task>_suite.pkl）")
    args = ap.parse_args()

    meta = RunMeta("evaluate_suite", vars(args))
    out_dir = ensure_dir(args.out or os.path.join(os.path.dirname(os.path.abspath(args.pred)), "eval"))
    inputs = None if args.no_align_check else (args.inputs or suite_io.input_path(args.task, args.mode))

    suite = suite_io.load_suite(args.task, path=args.suite)
    preds, confs = suite_io.read_predictions(args.task, args.pred)
    rows, n = suite_io.evaluate_to_dir(suite, args.task, args.mode, preds, confs, args.label, out_dir,
                                       inputs_path=inputs)
    meta.add(pred_file=file_info(args.pred), inputs_file=file_info(inputs), n_predictions=len(preds),
             n_tests=len(rows), n_testcase_records=n, out_dir=rel(out_dir))
    meta.save(out_dir)
    print("打分完成：%d 个测试，%d 条用例记录 -> %s" % (len(rows), n, rel(out_dir)))
    for r in rows:
        print("  [%-10s %-3s] %-70s fail %s%%" % (r["capability"], r["type"], r["test_name"][:70],
                                                  suite_io.fmt_rate(r["fail_rate"])))


if __name__ == "__main__":
    main()
