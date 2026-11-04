#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把多个 per_test.csv 并排成「原文表 1/2/3 式」的对照表（行 = 测试，列 = 模型失败率）。

两个环境都能运行（只用标准库）。第一列是基准（通常放作者公开预测），其余列额外给出与基准的差值。

  python code/common/make_table.py \
      --run "bert(released)=results/released_reference/sentiment/bert/per_test.csv" \
      --run "bert-textattack=results/sentiment/bert-textattack/released/eval/per_test.csv" \
      --out results/sentiment/table1_compare --title "表1 对照"

输出 <out>.csv 与 <out>.md。
"""
import argparse
import collections
import csv
import io
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.paths import ensure_dir  # noqa: E402


def read_rows(path):
    with io.open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build(runs, out_prefix, title="CheckList 失败率对照", with_delta=True):
    """runs: [(label, per_test_csv_path), ...]"""
    tables = [(label, read_rows(p)) for label, p in runs]
    order = collections.OrderedDict()
    for _, rows in tables:
        for r in rows:
            key = r["test_name"]
            if key not in order:
                order[key] = r
    by_label = {label: {r["test_name"]: r for r in rows} for label, rows in tables}
    labels = [label for label, _ in tables]
    base = labels[0]

    header = ["capability", "type", "test_name", "n_run"]
    for i, lab in enumerate(labels):
        header.append(lab)
        if with_delta and i > 0:
            header.append("Δ(%s−%s)" % (lab, base))
    out_rows = []
    for key, r0 in order.items():
        row = collections.OrderedDict([("capability", r0["capability"]), ("type", r0["type"]),
                                       ("test_name", key), ("n_run", r0.get("n_run", ""))])
        b = _f(by_label[base].get(key, {}).get("fail_rate"))
        for i, lab in enumerate(labels):
            v = _f(by_label[lab].get(key, {}).get("fail_rate"))
            row[lab] = "" if v is None else "%.1f" % v
            if with_delta and i > 0:
                row["Δ(%s−%s)" % (lab, base)] = "" if (v is None or b is None) else "%+.1f" % (v - b)
        out_rows.append(row)

    ensure_dir(os.path.dirname(os.path.abspath(out_prefix)))
    with io.open(out_prefix + ".csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    md = ["# %s" % title, "", "数值为失败率（%）。来源：" + "；".join("%s=`%s`" % (l, p) for l, p in runs), "",
          "| " + " | ".join(header) + " |", "|" + "|".join(["---"] * 3 + ["---:"] * (len(header) - 3)) + "|"]
    for r in out_rows:
        md.append("| " + " | ".join(str(r[h]).replace("|", "\\|") for h in header) + " |")
    with io.open(out_prefix + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    return out_rows


def main():
    ap = argparse.ArgumentParser(description="并排多个 per_test.csv")
    ap.add_argument("--run", action="append", required=True, help="label=path/to/per_test.csv，可重复")
    ap.add_argument("--out", required=True, help="输出前缀（不含扩展名）")
    ap.add_argument("--title", default="CheckList 失败率对照")
    ap.add_argument("--no-delta", action="store_true")
    args = ap.parse_args()
    runs = []
    for item in args.run:
        if "=" not in item:
            ap.error("--run 需要 label=path 形式：%s" % item)
        label, path = item.split("=", 1)
        runs.append((label, path))
    build(runs, args.out, args.title, with_delta=not args.no_delta)
    print("写出 %s.csv / %s.md" % (args.out, args.out))


if __name__ == "__main__":
    main()
