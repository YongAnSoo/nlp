#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""【环境 M｜成员 B】在 CheckList QQP 套件输入上跑模型，输出每行一个 P(duplicate)。

  python code/qqp/predict_qqp.py --model bert-textattack --mode released

输入：released = data/raw/release_data/qqp/tests_n500（tsv，表头 id/question1/question2）
      full     = data/generated/qqp/full_inputs.tsv（先在环境 E 跑 export_inputs.py）
输出：results/qqp/<model>/<mode>/{preds.txt, raw_probs.tsv, run_meta.json}
然后到环境 E：
  python code/common/evaluate_suite.py --task qqp --mode released \
      --pred results/qqp/bert-textattack/released/preds.txt --label bert-textattack
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common import hf_utils  # noqa: E402
from common.paths import RESULTS_DIR, ensure_dir, rel  # noqa: E402
from common.runmeta import RunMeta, file_info  # noqa: E402
from common.suite_io import input_path, read_input_file  # noqa: E402


def p_duplicate(entry, logits):
    kind = entry.get("kind", "binary")
    if kind == "binary":
        return hf_utils.softmax(logits)[:, int(entry.get("duplicate_index", 1))]
    if kind == "single_logit":
        return hf_utils.sigmoid(logits[:, 0])
    raise ValueError("未知 kind: %s" % kind)


def run_pairs(cfg, entry, pairs, device=None, batch_size=None, desc="qqp"):
    inf = cfg.get("inference", {})
    tok, model, dev, info = hf_utils.load_seq_cls(entry["source"], entry.get("revision"),
                                                  device or inf.get("device", "auto"))
    q1 = [a for a, _ in pairs]
    q2 = [b for _, b in pairs]
    logits, n_trunc = hf_utils.predict_logits(tok, model, dev, q1, q2, batch_size=batch_size or int(inf.get("batch_size", 64)),
                                              max_length=int(inf.get("max_length", 128)), desc=desc)
    return p_duplicate(entry, logits), logits, n_trunc, info


def main():
    ap = argparse.ArgumentParser(description="QQP 模型推理")
    ap.add_argument("--config", default="qqp")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="released", choices=["released", "full"])
    ap.add_argument("--inputs")
    ap.add_argument("--out")
    ap.add_argument("--device")
    ap.add_argument("--batch-size", type=int)
    args = ap.parse_args()

    cfg = hf_utils.load_config(args.config)
    entry = hf_utils.model_entry(cfg, args.model)
    hf_utils.set_seed(int(cfg.get("inference", {}).get("seed", 42)))
    meta = RunMeta("predict_qqp", {**vars(args), "inference": cfg.get("inference"), "model_entry": entry})
    inputs = args.inputs or input_path("qqp", args.mode)
    pairs = read_input_file("qqp", inputs)
    print("输入 %s：%d 对" % (rel(inputs), len(pairs)))
    out_dir = ensure_dir(args.out or os.path.join(RESULTS_DIR, "qqp", args.model, args.mode))

    p, logits, n_trunc, info = run_pairs(cfg, entry, pairs, args.device, args.batch_size)
    print("模型 %s | 标签 %s" % (entry["source"], hf_utils.describe_labels(info)))
    hf_utils.write_lines(os.path.join(out_dir, "preds.txt"), [repr(float(v)) for v in p])
    hf_utils.write_lines(os.path.join(out_dir, "raw_probs.tsv"),
                         ["line\t" + "\t".join("logit_%d" % i for i in range(logits.shape[1])) + "\tp_duplicate"] +
                         ["%d\t%s\t%r" % (i, "\t".join(repr(float(v)) for v in row), float(p[i]))
                          for i, row in enumerate(logits)])
    meta.add(model_info=info, inputs_file=file_info(inputs), n_examples=len(pairs), n_truncated=n_trunc,
             duplicate_rate_at_0_5=float((p >= 0.5).mean()), out_dir=rel(out_dir))
    meta.save(out_dir)
    print("P(dup)≥0.5 的比例 %.3f | 截断 %d | 写出 %s" % ((p >= 0.5).mean(), n_trunc, rel(out_dir)))


if __name__ == "__main__":
    main()
