#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""【环境 M｜成员 C】在 SQuAD v1.1 dev（10,570 条，全部）上确认模型 EM/F1 正常（基线指标）。

  python code/squad/validate_squad_dev.py --model bert-csarron

输出 results/squad/<model>/validation_squad/{predictions.jsonl, metrics.json, run_meta.json}
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from common import hf_utils  # noqa: E402
from common.paths import RESULTS_DIR, ensure_dir, rel  # noqa: E402
from common.runmeta import RunMeta  # noqa: E402
from predict_squad import answer_all, build_pipeline  # noqa: E402
import squad_metric  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="squad")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device")
    args = ap.parse_args()

    from datasets import load_dataset
    cfg = hf_utils.load_config(args.config)
    entry = hf_utils.model_entry(cfg, args.model)
    inf = cfg.get("inference", {})
    hf_utils.set_seed(int(inf.get("seed", 42)))
    v = cfg["validation"]
    meta = RunMeta("validate_squad_dev", {**vars(args), "model_entry": entry, "validation": v, "pipeline": cfg.get("pipeline")})
    try:
        ds = load_dataset(v["dataset"], split=v["split"])
    except Exception:
        ds = load_dataset("squad", split=v["split"])
    qa, info = build_pipeline(entry, args.device or inf.get("device", "auto"))
    res = answer_all(qa, list(ds["context"]), list(ds["question"]), cfg.get("pipeline", {}),
                     batch_size=int(inf.get("batch_size", 32)), chunk_size=int(inf.get("chunk_size", 2048)), desc="squad-dev")
    preds = [r["answer"] for r in res]
    refs = [a["text"] for a in ds["answers"]]
    m = squad_metric.evaluate(preds, refs)

    out_dir = ensure_dir(os.path.join(RESULTS_DIR, "squad", args.model, "validation_squad"))
    with open(os.path.join(out_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for i, (ex_id, p, r) in enumerate(zip(ds["id"], preds, refs)):
            f.write(json.dumps({"idx": i, "id": ex_id, "pred": p, "answers": r,
                                "em": squad_metric.metric_max_over_ground_truths(squad_metric.exact_match_score, p, r)},
                               ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    meta.add(model_info=info, metrics=m, out_dir=rel(out_dir))
    meta.save(out_dir)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
