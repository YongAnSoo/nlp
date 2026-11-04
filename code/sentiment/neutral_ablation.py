#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""【成员 A 的消融】只改变「中性阈值」margin，其余全部固定，看失败率怎么变。

不需要重新跑模型：直接读已保存的 P(positive)（.npy 或 predict_sentiment.py 写的 raw_probs.tsv），
对每个 margin 生成 checklist 预测文件，再逐个打分并并排成表。两步都只依赖 numpy + checklist，
所以可以全部在「环境 E（Python 3.7）」里完成（例如 A 在 Kaggle 上建好的 checklist-py37）。
保持 Python 3.6 兼容。

第 1 步：生成各 margin 的预测文件
  python code/sentiment/neutral_ablation.py make \
      --p-pos /kaggle/working/member_a_results/bert_own_positive_probs.npy --label bert-textattack \
      --out results/sentiment/bert-textattack/released/neutral_ablation

第 2 步：打分（环境 E）
  python code/sentiment/neutral_ablation.py score \
      --dir results/sentiment/bert-textattack/released/neutral_ablation \
      --suite /kaggle/working/member_a_results/reference/sentiment_suite.pkl \
      --inputs /kaggle/working/member_a_results/reference/tests_n500

输出（--dir 下）：
  preds_margin<m>.txt        各 margin 的预测文件
  manifest.json              每个 margin 的负/中/正数量、文件 sha256、P(pos) 来源文件 sha256
  margin<m>/per_test.csv ... 各 margin 的完整打分结果（同 evaluate_suite.py）
  ablation_table.{csv,md}    38 项测试 × 各 margin 的失败率；第一列是原文设置 margin=1/3，其余列给出差值
"""
import argparse
import io
import json
import os
import sys

import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from common.paths import ensure_dir, rel  # noqa: E402
from common.runmeta import RunMeta, file_info  # noqa: E402
from neutral import PAPER_MARGIN, binary_to_three_way, format_pred_lines  # noqa: E402

DEFAULT_MARGINS = [0.0, 0.1, 0.2, PAPER_MARGIN, 0.4, 0.5]


def tag(m):
    return "%.3f" % m


def load_p_pos(path, column=None):
    """读 P(positive)：.npy（一维）或 raw_probs.tsv（指定列名，默认取最后一列）。"""
    if path.endswith(".npy"):
        p = np.load(path, allow_pickle=False)
    else:
        with io.open(path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            col = header.index(column) if column else len(header) - 1
            p = np.array([float(line.rstrip("\n").split("\t")[col]) for line in f])
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
        raise ValueError("P(positive) 必须是 [0,1] 内的有限数")
    return p


def cmd_make(args):
    meta = RunMeta("neutral_ablation.make", vars(args))
    p = load_p_pos(args.p_pos, args.column)
    out = ensure_dir(args.out)
    margins = [float(m) for m in (args.margins or DEFAULT_MARGINS)]
    entries = []
    for m in margins:
        preds, probs = binary_to_three_way(p, m)
        path = os.path.join(out, "preds_margin%s.txt" % tag(m))
        with io.open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(format_pred_lines(preds, probs)) + "\n")
        counts = np.bincount(preds, minlength=3).tolist()
        entries.append({"margin": m, "file": os.path.basename(path), "counts_neg_neu_pos": counts,
                        "sha256": file_info(path)["sha256"]})
        print("margin=%s  负/中/正 = %s  -> %s" % (tag(m), counts, rel(path)))
    manifest = {"label": args.label, "n": int(p.shape[0]), "p_pos_source": file_info(args.p_pos),
                "paper_margin": PAPER_MARGIN, "margins": entries}
    with io.open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    meta.add(manifest=manifest)
    meta.save(out, "make_run_meta.json")


def cmd_score(args):
    from common import make_table, suite_io
    with io.open(os.path.join(args.dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    meta = RunMeta("neutral_ablation.score", vars(args))
    # 原文设置（1/3）放第一列作基准，其余按 margin 升序
    entries = sorted(manifest["margins"], key=lambda e: (abs(e["margin"] - PAPER_MARGIN) > 1e-9, e["margin"]))
    runs = []
    for e in entries:
        suite = suite_io.load_suite("sentiment", path=args.suite, strict_python=not args.no_strict_python)
        preds, confs = suite_io.read_predictions("sentiment", os.path.join(args.dir, e["file"]))
        out_dir = os.path.join(args.dir, "margin%s" % tag(e["margin"]))
        label = "margin=%s" % tag(e["margin"])
        rows, n = suite_io.evaluate_to_dir(suite, "sentiment", "released", preds, confs,
                                           "%s|%s" % (manifest["label"], label), out_dir, inputs_path=args.inputs)
        runs.append((label, os.path.join(out_dir, "per_test.csv")))
        print("%s: %d 个测试已打分 -> %s" % (label, len(rows), rel(out_dir)))
    make_table.build(runs, os.path.join(args.dir, "ablation_table"),
                     title="中性阈值消融：%s（基准列 = 原文设置 margin=1/3）" % manifest["label"])
    meta.add(n_margins=len(runs), inputs_file=file_info(args.inputs) if args.inputs else None,
             suite_file=file_info(args.suite) if args.suite else None)
    meta.save(args.dir, "score_run_meta.json")
    print("并排表:", rel(os.path.join(args.dir, "ablation_table.md")))


def main():
    ap = argparse.ArgumentParser(description="中性阈值消融")
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("make", help="由 P(positive) 生成各 margin 的预测文件")
    a.add_argument("--p-pos", required=True, help=".npy 或 raw_probs.tsv")
    a.add_argument("--column", help="raw_probs.tsv 中 P(positive) 的列名（默认最后一列）")
    a.add_argument("--label", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--margins", type=float, nargs="+", help="默认 0 0.1 0.2 1/3 0.4 0.5")
    b = sub.add_parser("score", help="【环境 E】对 make 生成的各文件打分并并排")
    b.add_argument("--dir", required=True, help="make 的 --out 目录")
    b.add_argument("--suite", help="sentiment_suite.pkl（默认 data/raw/release_data/...）")
    b.add_argument("--inputs", help="tests_n500，用于逐行对齐校验（默认 data/raw/release_data/...）")
    b.add_argument("--no-strict-python", action="store_true", help=argparse.SUPPRESS)  # 仅供单元测试
    args = ap.parse_args()
    if args.cmd == "make":
        cmd_make(args)
    elif args.cmd == "score":
        if not args.inputs:
            from common import suite_io
            args.inputs = suite_io.input_path("sentiment", "released")
        cmd_score(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
