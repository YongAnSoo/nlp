#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""【环境 M｜成员 C】用 HF 的 SQuAD 模型跑 CheckList 阅读理解套件，每行输出一个答案片段。

  python code/squad/predict_squad.py --model bert-csarron --mode released

输入：released = data/raw/release_data/squad/squad.jsonl（{"passage","question"}，71,292 行）
      full     = data/generated/squad/full_inputs.jsonl（先在环境 E 跑 export_inputs.py）
输出：results/squad/<model>/<mode>/{preds.txt, raw_answers.jsonl, run_meta.json}
然后到环境 E：
  python code/common/evaluate_suite.py --task squad --mode released \
      --pred results/squad/bert-csarron/released/preds.txt --label bert-csarron
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common import hf_utils  # noqa: E402
from common.paths import RESULTS_DIR, ensure_dir, rel  # noqa: E402
from common.runmeta import RunMeta, file_info  # noqa: E402
from common.suite_io import input_path, read_input_file  # noqa: E402


def build_pipeline(entry, device="auto"):
    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
    tok = AutoTokenizer.from_pretrained(entry["source"], revision=entry.get("revision"))
    model = AutoModelForQuestionAnswering.from_pretrained(entry["source"], revision=entry.get("revision"))
    dev = hf_utils.pick_device(device)
    qa = pipeline("question-answering", model=model, tokenizer=tok, device=dev)
    info = {"source": entry["source"], "revision_requested": entry.get("revision"),
            "commit_hash": getattr(model.config, "_commit_hash", None), "device": str(dev),
            "torch_threads": torch.get_num_threads()}
    return qa, info


def answer_all(qa, contexts, questions, pipe_cfg, batch_size=32, chunk_size=2048, desc="squad"):
    """返回与输入等长的 [{"answer","score","start","end"}]。"""
    from tqdm.auto import tqdm
    kwargs = {k: pipe_cfg[k] for k in ("max_answer_len", "max_seq_len", "doc_stride", "max_question_len",
                                       "handle_impossible_answer") if k in pipe_cfg}
    out = []
    for s in tqdm(range(0, len(questions), chunk_size), desc=desc):
        q, c = questions[s:s + chunk_size], contexts[s:s + chunk_size]
        res = qa(question=q, context=c, batch_size=batch_size, top_k=1, **kwargs)
        if isinstance(res, dict):
            res = [res]
        out.extend(res)
    assert len(out) == len(questions), (len(out), len(questions))
    return out


def main():
    ap = argparse.ArgumentParser(description="SQuAD 模型推理")
    ap.add_argument("--config", default="squad")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="released", choices=["released", "full"])
    ap.add_argument("--inputs")
    ap.add_argument("--out")
    ap.add_argument("--device")
    ap.add_argument("--batch-size", type=int)
    args = ap.parse_args()

    cfg = hf_utils.load_config(args.config)
    entry = hf_utils.model_entry(cfg, args.model)
    inf = dict(cfg.get("inference", {}))
    hf_utils.set_seed(int(inf.get("seed", 42)))
    meta = RunMeta("predict_squad", {**vars(args), "inference": inf, "pipeline": cfg.get("pipeline"), "model_entry": entry})
    inputs = args.inputs or input_path("squad", args.mode)
    items = read_input_file("squad", inputs)
    print("输入 %s：%d 条" % (rel(inputs), len(items)))
    out_dir = ensure_dir(args.out or os.path.join(RESULTS_DIR, "squad", args.model, args.mode))

    qa, info = build_pipeline(entry, args.device or inf.get("device", "auto"))
    res = answer_all(qa, [c for c, _ in items], [q for _, q in items], cfg.get("pipeline", {}),
                     batch_size=args.batch_size or int(inf.get("batch_size", 32)),
                     chunk_size=int(inf.get("chunk_size", 2048)))
    # checklist 的 pred_only 格式按行读取：答案里的换行替换成空格（数目写进 run_meta）
    n_newline = sum("\n" in r["answer"] or "\r" in r["answer"] for r in res)
    hf_utils.write_lines(os.path.join(out_dir, "preds.txt"),
                         [r["answer"].replace("\r", " ").replace("\n", " ") for r in res])
    with open(os.path.join(out_dir, "raw_answers.jsonl"), "w", encoding="utf-8") as f:
        for i, r in enumerate(res):
            f.write(json.dumps({"line": i, "answer": r["answer"], "score": float(r["score"]),
                                "start": int(r["start"]), "end": int(r["end"])}, ensure_ascii=False) + "\n")
    meta.add(model_info=info, inputs_file=file_info(inputs), n_examples=len(items),
             n_answers_with_newline=n_newline, out_dir=rel(out_dir))
    meta.save(out_dir)
    print("写出", rel(out_dir))


if __name__ == "__main__":
    main()
