#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 per_example.jsonl.gz 导出错误案例，供报告第 4 节与附录（≥10 例）使用。只用标准库。

输出：
  failures_all.csv       全部失败用例（不抽样）
  failure_counts.csv     按 capability / test 统计的失败数
  cases_for_report.csv   按测试轮流抽取的 N 个案例（默认 12，固定种子），带空白列「错误类型」「解释」待人工填写；
                         若失败用例不足 N，用「通过但期望值最接近 0」的易混淆用例补足（标注 passed=True）

  python code/common/export_failures.py \
      --per-example results/squad/bert-csarron/released/eval/per_example.jsonl.gz \
      --compare roberta=results/squad/roberta-deepset/released/eval/per_example.jsonl.gz \
      --capability Coref --capability Negation --n-sample 12 --seed 0
"""
import argparse
import collections
import csv
import gzip
import io
import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.paths import ensure_dir, rel  # noqa: E402


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def render_example(task, ex):
    if task == "squad" and isinstance(ex, list) and len(ex) == 2 and all(isinstance(s, str) for s in ex):
        return "C: %s\nQ: %s" % (ex[0], ex[1])
    if task == "qqp" and isinstance(ex, list) and len(ex) == 2 and all(isinstance(s, str) for s in ex):
        return "%s ||| %s" % (ex[0], ex[1])
    if isinstance(ex, str):
        return ex
    return json.dumps(ex, ensure_ascii=False)


def render_examples(task, examples):
    """单样本用例直接渲染；多样本用例（INV/DIR 的原句 + 扰动句）逐个编号。"""
    is_single = isinstance(examples, str) or (
        task in ("qqp", "squad") and isinstance(examples, list) and len(examples) == 2
        and all(isinstance(s, str) for s in examples))
    if is_single:
        return render_example(task, examples)
    return "\n".join("[%d] %s" % (i, render_example(task, e)) for i, e in enumerate(examples))


def min_expect(rec):
    vals = rec.get("expect_results")
    if not isinstance(vals, list):
        vals = [vals]
    nums = [float(v) for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    bools = [v for v in vals if isinstance(v, bool)]
    if nums:
        return min(nums)
    return 1.0 if all(bools) else 0.0


FIELDS = ["id", "task", "model", "capability", "type", "test_name", "testcase_idx", "input_lines",
          "input", "label", "preds", "confs", "passed"]


def row_of(rec, compare=None):
    row = collections.OrderedDict()
    for k in FIELDS:
        v = rec.get(k)
        if k == "input":
            v = render_examples(rec["task"], rec["examples"])
        elif isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False)
        row[k] = v
    for label, recs in (compare or {}).items():
        other = recs.get(rec["id"])
        row["preds[%s]" % label] = "" if other is None else json.dumps(other["preds"], ensure_ascii=False)
        row["passed[%s]" % label] = "" if other is None else other["passed"]
    return row


def write_csv(rows, path, extra=()):
    fields = list(rows[0].keys()) + list(extra) if rows else FIELDS + list(extra)
    with io.open(path, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig：Excel 打开不乱码
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            r = dict(r)
            for e in extra:
                r.setdefault(e, "")
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="导出错误案例")
    ap.add_argument("--per-example", required=True)
    ap.add_argument("--compare", action="append", default=[], help="label=另一个模型的 per_example.jsonl.gz")
    ap.add_argument("--out", help="输出目录（默认与 per_example 同目录下 cases/）")
    ap.add_argument("--n-sample", type=int, default=12, help="报告用案例数（课程要求附录≥10）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--capability", action="append", help="只从这些 capability 中抽报告案例（全部失败仍导出）")
    args = ap.parse_args()

    out_dir = ensure_dir(args.out or os.path.join(os.path.dirname(os.path.abspath(args.per_example)), "cases"))
    recs = list(load(args.per_example))
    compare = {}
    for item in args.compare:
        label, path = item.split("=", 1)
        compare[label] = {r["id"]: r for r in load(path)}

    fails = [r for r in recs if r["passed"] is False]
    write_csv([row_of(r, compare) for r in fails], os.path.join(out_dir, "failures_all.csv"))

    counts = collections.OrderedDict()
    for r in recs:
        key = (r["capability"], r["type"], r["test_name"])
        c = counts.setdefault(key, {"n_cases": 0, "n_filtered": 0, "n_fail": 0})
        c["n_cases"] += 1
        c["n_filtered"] += r["passed"] is None
        c["n_fail"] += r["passed"] is False
    with io.open(os.path.join(out_dir, "failure_counts.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["capability", "type", "test_name", "n_cases", "n_filtered", "n_fail"])
        for (cap, typ, name), c in counts.items():
            w.writerow([cap, typ, name, c["n_cases"], c["n_filtered"], c["n_fail"]])

    # ---- 报告用抽样：按测试轮流取，保证覆盖面 ----
    rng = random.Random(args.seed)
    pool = [r for r in fails if not args.capability or r["capability"] in args.capability]
    by_test = collections.OrderedDict()
    for r in sorted(pool, key=lambda r: r["test_name"]):
        by_test.setdefault(r["test_name"], []).append(r)
    for lst in by_test.values():
        rng.shuffle(lst)
    chosen = []
    while len(chosen) < args.n_sample and any(by_test.values()):
        for name in list(by_test):
            if by_test[name] and len(chosen) < args.n_sample:
                chosen.append(by_test[name].pop())
    if len(chosen) < args.n_sample:  # 失败不足时，补「易混淆但通过」的用例
        passed = [r for r in recs if r["passed"] is True
                  and (not args.capability or r["capability"] in args.capability)]
        passed.sort(key=min_expect)
        chosen += passed[: args.n_sample - len(chosen)]
    write_csv([row_of(r, compare) for r in chosen], os.path.join(out_dir, "cases_for_report.csv"),
              extra=("错误类型", "解释（人工填写）"))

    print("全部用例 %d，失败 %d（已全部写入 failures_all.csv）" % (len(recs), len(fails)))
    print("报告案例 %d 条 -> %s" % (len(chosen), rel(os.path.join(out_dir, "cases_for_report.csv"))))
    by_cap = collections.Counter(r["capability"] for r in fails)
    for cap, n in by_cap.most_common():
        print("  %-12s 失败 %d" % (cap, n))


if __name__ == "__main__":
    main()
