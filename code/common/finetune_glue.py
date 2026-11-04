#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""【环境 M｜成员 B（也可给 A 用）】在 GLUE QQP / SST-2 上微调 BERT / RoBERTa，用于没有现成权重时。

  python code/common/finetune_glue.py --task qqp --model roberta-base --seed 42
  python code/common/finetune_glue.py --task qqp --model roberta-base --seed 43
  python code/common/finetune_glue.py --task qqp --model roberta-base --seed 44

* 只在训练集上训练（默认全部训练样本），在验证集上选 epoch；不碰测试集。
* 超参数默认取 configs/qqp.yaml 的 finetune 段；命令行可覆盖。
* 权重存到 checkpoints/<task>/<model>-seed<seed>/best（不入库）；指标与日志存到
  results/<task>/finetune/<model>-seed<seed>/。
"""
import argparse
import inspect
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common import hf_utils  # noqa: E402
from common.paths import CHECKPOINT_DIR, RESULTS_DIR, ensure_dir, rel  # noqa: E402
from common.runmeta import RunMeta  # noqa: E402

TEXT_FIELDS = {"qqp": ("question1", "question2"), "sst2": ("sentence", None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TEXT_FIELDS))
    ap.add_argument("--model", required=True, help="如 bert-base-uncased / roberta-base")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default=None, help="默认 configs/<task 对应>.yaml（qqp -> qqp, sst2 -> sentiment）")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--epochs", type=float)
    ap.add_argument("--max-length", type=int)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments)

    cfg = hf_utils.load_config(args.config or ("qqp" if args.task == "qqp" else "sentiment"))
    ft = dict(cfg.get("finetune", {}))
    lr = args.lr or float(ft.get("learning_rate", 2e-5))
    bs = args.batch_size or int(ft.get("batch_size", 32))
    epochs = args.epochs or float(ft.get("epochs", 3))
    max_len = args.max_length or int(ft.get("max_length", 128))
    hf_utils.set_seed(args.seed)

    name = "%s-seed%d" % (args.model.replace("/", "_"), args.seed)
    ckpt_dir = ensure_dir(os.path.join(CHECKPOINT_DIR, args.task, name))
    out_dir = ensure_dir(os.path.join(RESULTS_DIR, args.task if args.task == "qqp" else "sentiment", "finetune", name))
    meta = RunMeta("finetune_glue", {**vars(args), "lr": lr, "batch_size": bs, "epochs": epochs, "max_length": max_len,
                                     "warmup_ratio": ft.get("warmup_ratio", 0.06), "weight_decay": ft.get("weight_decay", 0.01)})

    try:
        raw = load_dataset("nyu-mll/glue", args.task)
    except Exception:
        raw = load_dataset("glue", args.task)
    a, b = TEXT_FIELDS[args.task]
    tok = AutoTokenizer.from_pretrained(args.model)

    def enc(batch):
        return tok(batch[a], batch[b] if b else None, truncation=True, max_length=max_len)
    data = raw.map(enc, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    def metrics(p):
        pred = np.argmax(p.predictions, axis=1)
        y = p.label_ids
        tp, fp, fn = ((pred == 1) & (y == 1)).sum(), ((pred == 1) & (y == 0)).sum(), ((pred == 0) & (y == 1)).sum()
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        return {"accuracy": float((pred == y).mean()), "f1": float(f1)}

    kw = dict(output_dir=ckpt_dir, learning_rate=lr, per_device_train_batch_size=bs, per_device_eval_batch_size=bs * 2,
              num_train_epochs=epochs, warmup_ratio=float(ft.get("warmup_ratio", 0.06)),
              weight_decay=float(ft.get("weight_decay", 0.01)), seed=args.seed, data_seed=args.seed,
              save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="accuracy",
              save_total_limit=1, logging_steps=200, report_to=[], fp16=args.fp16)
    params = inspect.signature(TrainingArguments.__init__).parameters
    kw["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "epoch"
    targs = TrainingArguments(**kw)
    trainer = Trainer(model=model, args=targs, train_dataset=data["train"], eval_dataset=data["validation"],
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    train_out = trainer.train()
    final = trainer.evaluate()
    best_dir = os.path.join(ckpt_dir, "best")
    trainer.save_model(best_dir)
    tok.save_pretrained(best_dir)

    result = {"train": train_out.metrics, "validation": final, "n_train": len(data["train"]),
              "n_validation": len(data["validation"]), "best_checkpoint": rel(best_dir),
              "log_history": trainer.state.log_history}
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    meta.add(n_train=len(data["train"]), n_validation=len(data["validation"]), validation=final, best_checkpoint=rel(best_dir))
    meta.save(out_dir)
    print("验证集:", final, "\n权重:", rel(best_dir), "\n指标:", rel(out_dir))


if __name__ == "__main__":
    main()
