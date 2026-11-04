#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""【环境 M｜成员 A】在 SST-2 验证集（872 条，全部）上确认模型准确率正常，并检查标签方向。

这是 CheckList 对照的「基线指标」：held-out accuracy 高，不代表行为测试失败率低。

  python code/sentiment/validate_sst2.py --model bert-textattack

输出 results/sentiment/<model>/validation_sst2/{predictions.tsv, metrics.json, run_meta.json}
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common import hf_utils  # noqa: E402
from common.paths import RESULTS_DIR, ensure_dir, rel  # noqa: E402
from common.runmeta import RunMeta  # noqa: E402


def load_split(v):
    from datasets import load_dataset
    try:
        return load_dataset(v["dataset"], v.get("config"), split=v["split"])
    except Exception as e:  # 旧版 datasets 用 "glue"
        print("[提示] 加载 %s 失败（%s），改用 glue" % (v["dataset"], e))
        return load_dataset("glue", v.get("config"), split=v["split"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="sentiment")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device")
    args = ap.parse_args()

    cfg = hf_utils.load_config(args.config)
    entry = hf_utils.model_entry(cfg, args.model)
    inf = cfg.get("inference", {})
    hf_utils.set_seed(int(inf.get("seed", 42)))
    meta = RunMeta("validate_sst2", {**vars(args), "model_entry": entry, "validation": cfg["validation"]})
    ds = load_split(cfg["validation"])
    texts, labels = list(ds["sentence"]), np.array(ds["label"])
    tok, model, dev, info = hf_utils.load_seq_cls(entry["source"], entry.get("revision"), args.device or inf.get("device", "auto"))
    logits, n_trunc = hf_utils.predict_logits(tok, model, dev, texts, batch_size=int(inf.get("batch_size", 64)),
                                              max_length=int(inf.get("max_length", 128)), desc="sst2-dev")
    probs = hf_utils.softmax(logits)
    if entry.get("kind", "binary") == "binary":
        p_pos = probs[:, int(entry["positive_index"])]
        p_neg = 1 - p_pos
    else:  # 三分类模型：只比较正负两类
        li = entry["label_indices"]
        p_pos, p_neg = probs[:, int(li["positive"])], probs[:, int(li["negative"])]
    pred = (p_pos >= p_neg).astype(int)
    acc = float((pred == labels).mean())
    acc_flipped = float(((1 - pred) == labels).mean())

    out_dir = ensure_dir(os.path.join(RESULTS_DIR, "sentiment", args.model, "validation_sst2"))
    hf_utils.write_lines(os.path.join(out_dir, "predictions.tsv"),
                         ["idx\tlabel\tpred\tp_pos\tsentence"] +
                         ["%d\t%d\t%d\t%r\t%s" % (i, labels[i], pred[i], float(p_pos[i]), texts[i].replace("\t", " "))
                          for i in range(len(texts))])
    metrics = {"n": int(len(labels)), "accuracy": acc, "accuracy_if_labels_flipped": acc_flipped,
               "n_truncated": n_trunc, "positive_rate_pred": float(pred.mean()), "positive_rate_gold": float(labels.mean())}
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    meta.add(model_info=info, metrics=metrics, out_dir=rel(out_dir))
    meta.save(out_dir)
    print(json.dumps(metrics, indent=2))
    if acc_flipped > acc:
        print("[警告] 翻转标签后准确率更高：positive_index 很可能设反了，请检查 configs/sentiment.yaml")


if __name__ == "__main__":
    main()
