#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""【环境 M｜成员 B】先确认 QQP 模型在 GLUE QQP 验证集（40,430 对，全部）上准确率正常。

与 CheckList 失败率放在一起，就是「准确率高、失败率也高」的对照。

  python code/qqp/validate_qqp_dev.py --model bert-textattack

输出 results/qqp/<model>/validation_qqp/{predictions.tsv, metrics.json, run_meta.json}
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from common import hf_utils  # noqa: E402
from common.paths import RESULTS_DIR, ensure_dir, rel  # noqa: E402
from common.runmeta import RunMeta  # noqa: E402
from predict_qqp import run_pairs  # noqa: E402


def load_split(v):
    from datasets import load_dataset
    try:
        return load_dataset(v["dataset"], v.get("config"), split=v["split"])
    except Exception as e:
        print("[提示] 加载 %s 失败（%s），改用 glue" % (v["dataset"], e))
        return load_dataset("glue", v.get("config"), split=v["split"])


def binary_metrics(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": float((pred == y).mean()), "precision": prec, "recall": rec, "f1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qqp")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device")
    args = ap.parse_args()

    cfg = hf_utils.load_config(args.config)
    entry = hf_utils.model_entry(cfg, args.model)
    hf_utils.set_seed(int(cfg.get("inference", {}).get("seed", 42)))
    v = cfg["validation"]
    meta = RunMeta("validate_qqp_dev", {**vars(args), "model_entry": entry, "validation": v})
    ds = load_split(v)
    pairs = list(zip(ds["question1"], ds["question2"]))
    y = np.array(ds["label"])
    p, _, n_trunc, info = run_pairs(cfg, entry, pairs, args.device, desc="qqp-dev")
    thr = float(v.get("threshold", 0.5))
    pred = (p >= thr).astype(int)
    m = binary_metrics(y, pred)
    m.update({"n": int(len(y)), "threshold": thr, "n_truncated": n_trunc,
              "accuracy_if_labels_flipped": float(((1 - pred) == y).mean())})

    out_dir = ensure_dir(os.path.join(RESULTS_DIR, "qqp", args.model, "validation_qqp"))
    hf_utils.write_lines(os.path.join(out_dir, "predictions.tsv"),
                         ["idx\tlabel\tpred\tp_duplicate\tquestion1\tquestion2"] +
                         ["%d\t%d\t%d\t%r\t%s\t%s" % (i, y[i], pred[i], float(p[i]),
                                                      pairs[i][0].replace("\t", " "), pairs[i][1].replace("\t", " "))
                          for i in range(len(y))])
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    meta.add(model_info=info, metrics=m, out_dir=rel(out_dir))
    meta.save(out_dir)
    print(json.dumps(m, indent=2))
    if m["accuracy_if_labels_flipped"] > m["accuracy"]:
        print("[警告] 翻转后准确率更高：duplicate_index 可能设反了")


if __name__ == "__main__":
    main()
