#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""【环境 M｜成员 A】在 CheckList 情感套件的输入上跑 HF 模型，写出 checklist 格式的预测文件。

  python code/sentiment/predict_sentiment.py --model bert-textattack --mode released
  python code/sentiment/predict_sentiment.py --model bert-textattack --mode full

输入：released = data/raw/release_data/sentiment/tests_n500（作者的抽样，每个测试至多 500 个用例）
      full     = data/generated/sentiment/full_inputs.txt（全部用例，先在环境 E 跑 export_inputs.py）
输出：results/sentiment/<model>/<mode>/
  raw_probs.tsv               每行一个样本的完整 softmax（不做任何舍入），用于之后改阈值而不必重跑模型
  preds.txt                   checklist 格式 "pred p_neg p_neu p_pos"，中性阈值取配置里的 neutral_margin
  preds_margin<m>.txt         消融：其它中性阈值（只对 binary 模型）
  run_meta.json               运行证据
然后到环境 E 打分：
  python code/common/evaluate_suite.py --task sentiment --mode released \
      --pred results/sentiment/bert-textattack/released/preds.txt --label bert-textattack
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from common import hf_utils  # noqa: E402
from common.paths import RESULTS_DIR, ensure_dir, rel  # noqa: E402
from common.runmeta import RunMeta, file_info  # noqa: E402
from common.suite_io import input_path, read_input_file  # noqa: E402
from neutral import binary_to_three_way, format_pred_lines  # noqa: E402


def margin_tag(m):
    return "%.3f" % m


def to_three_way(entry, probs, margin):
    kind = entry.get("kind", "binary")
    if kind == "binary":
        p_pos = probs[:, int(entry["positive_index"])]
        return binary_to_three_way(p_pos, margin)
    if kind == "three_way":
        li = entry["label_indices"]
        p3 = probs[:, [int(li["negative"]), int(li["neutral"]), int(li["positive"])]]
        return p3.argmax(axis=1), p3
    raise ValueError("未知 kind: %s" % kind)


def main():
    ap = argparse.ArgumentParser(description="情感模型推理")
    ap.add_argument("--config", default="sentiment")
    ap.add_argument("--model", required=True, help="configs/sentiment.yaml 里 models 下的键")
    ap.add_argument("--mode", default="released", choices=["released", "full"])
    ap.add_argument("--inputs", help="自定义输入文件（默认按 --mode）")
    ap.add_argument("--out", help="输出目录（默认 results/sentiment/<model>/<mode>）")
    ap.add_argument("--device", help="覆盖配置里的 device")
    ap.add_argument("--batch-size", type=int)
    args = ap.parse_args()

    cfg = hf_utils.load_config(args.config)
    entry = hf_utils.model_entry(cfg, args.model)
    inf = dict(cfg.get("inference", {}))
    if args.device:
        inf["device"] = args.device
    if args.batch_size:
        inf["batch_size"] = args.batch_size
    hf_utils.set_seed(int(inf.get("seed", 42)))
    meta = RunMeta("predict_sentiment", {**vars(args), "inference": inf, "model_entry": entry})

    inputs = args.inputs or input_path("sentiment", args.mode)
    texts = read_input_file("sentiment", inputs)
    print("输入 %s：%d 条" % (rel(inputs), len(texts)))
    out_dir = ensure_dir(args.out or os.path.join(RESULTS_DIR, "sentiment", args.model, args.mode))

    tok, model, dev, info = hf_utils.load_seq_cls(entry["source"], entry.get("revision"), inf.get("device", "auto"))
    print("模型 %s | 设备 %s | 标签 %s" % (entry["source"], dev, hf_utils.describe_labels(info)))
    logits, n_trunc = hf_utils.predict_logits(tok, model, dev, texts, batch_size=int(inf.get("batch_size", 64)),
                                              max_length=int(inf.get("max_length", 128)), desc="sentiment")
    probs = hf_utils.softmax(logits)

    # 原始概率：完整精度，便于复查与改阈值
    header = "\t".join(["line"] + ["p_%s" % (info["id2label"] or {}).get(i, i) for i in range(probs.shape[1])])
    hf_utils.write_lines(os.path.join(out_dir, "raw_probs.tsv"),
                         [header] + ["%d\t%s" % (i, "\t".join(repr(float(v)) for v in row)) for i, row in enumerate(probs)])

    margin = float(cfg.get("neutral_margin", 1 / 3))
    preds, p3 = to_three_way(entry, probs, margin)
    hf_utils.write_lines(os.path.join(out_dir, "preds.txt"), format_pred_lines(preds, p3))
    written = {"preds.txt": {"margin": margin, "label_counts": np.bincount(preds, minlength=3).tolist()}}
    if entry.get("kind", "binary") == "binary":
        for m in cfg.get("ablation_margins", []):
            m = float(m)
            pr, pp = binary_to_three_way(probs[:, int(entry["positive_index"])], m)
            name = "preds_margin%s.txt" % margin_tag(m)
            hf_utils.write_lines(os.path.join(out_dir, name), format_pred_lines(pr, pp))
            written[name] = {"margin": m, "label_counts": np.bincount(pr, minlength=3).tolist()}

    meta.add(model_info=info, inputs_file=file_info(inputs), n_examples=len(texts), n_truncated=n_trunc,
             outputs=written, out_dir=rel(out_dir))
    meta.save(out_dir)
    print("预测分布（负/中/正）:", written["preds.txt"]["label_counts"], "| 截断样本:", n_trunc)
    print("写出", rel(out_dir))
    print("下一步（环境 E）：python code/common/evaluate_suite.py --task sentiment --mode %s --pred %s --label %s"
          % (args.mode, rel(os.path.join(out_dir, "preds.txt")), args.model))


if __name__ == "__main__":
    main()
