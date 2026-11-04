#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""【环境 M｜成员 A，可选】用 OpenAI 兼容接口（如 DeepSeek）的大模型做三分类，输出 checklist 预测文件。

API Key 只从环境变量读取，绝不写入代码、配置或结果文件：
  export LLM_API_BASE=https://api.deepseek.com      # 或其它 OpenAI 兼容地址
  export LLM_MODEL=deepseek-chat
  export LLM_API_KEY=...                            # 不要提交到 git
  python code/sentiment/predict_sentiment_api.py --mode released

注意：
  * released 输入共 87,470 行（去重后约 8.4 万条），调用量与费用请先估算；脚本按文本去重并缓存，
    中断后重跑只会补齐缺失的部分。课程不要求自费，若只跑子集，必须在报告中写明范围，不能当作完整结果。
  * 大模型只给出离散标签，这里把置信度记为 one-hot（所选类 = 1.0），因此 DIR 测试里
    「置信度单调变化」类的期望会退化，解读时要说明这一限制。
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common import hf_utils  # noqa: E402
from common.paths import RESULTS_DIR, ensure_dir, rel  # noqa: E402
from common.runmeta import RunMeta, file_info  # noqa: E402
from common.suite_io import input_path, read_input_file  # noqa: E402

PROMPT = ("Classify the sentiment of the following text as negative, neutral, or positive. "
          "Answer with exactly one word: negative, neutral, or positive.\n\nText: {text}")
LABELS = {"negative": 0, "neutral": 1, "positive": 2}


def key_of(model, text):
    return hashlib.sha256((model + "\x00" + PROMPT + "\x00" + text).encode("utf-8")).hexdigest()


def call(base, key, model, text, temperature, retries=5):
    url = base.rstrip("/") + "/chat/completions"
    body = {"model": model, "temperature": temperature, "max_tokens": 5,
            "messages": [{"role": "user", "content": PROMPT.format(text=text)}]}
    for attempt in range(retries):
        try:
            r = requests.post(url, headers={"Authorization": "Bearer " + key}, json=body, timeout=60)
            if r.status_code == 200:
                d = r.json()
                return d["choices"][0]["message"]["content"], d.get("usage", {})
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError("HTTP %d: %s" % (r.status_code, r.text[:200]))
        except requests.RequestException:
            time.sleep(2 ** attempt)
    raise RuntimeError("多次重试仍失败")


def parse(answer):
    a = (answer or "").strip().lower()
    for name, idx in LABELS.items():
        if a.startswith(name):
            return idx
    for name, idx in LABELS.items():
        if name in a:
            return idx
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="sentiment")
    ap.add_argument("--mode", default="released", choices=["released", "full"])
    ap.add_argument("--label", default="llm-api")
    ap.add_argument("--max-workers", type=int)
    args = ap.parse_args()

    cfg = hf_utils.load_config(args.config)["api"]
    base, model, key = os.environ.get(cfg["base_url_env"]), os.environ.get(cfg["model_env"]), os.environ.get("LLM_API_KEY")
    if not (base and model and key):
        sys.exit("请设置环境变量 %s / %s / LLM_API_KEY" % (cfg["base_url_env"], cfg["model_env"]))
    meta = RunMeta("predict_sentiment_api", {"mode": args.mode, "api_base": base, "api_model": model,
                                             "prompt": PROMPT, "temperature": cfg["temperature"]})
    inputs = input_path("sentiment", args.mode)
    texts = read_input_file("sentiment", inputs)
    out_dir = ensure_dir(os.path.join(RESULTS_DIR, "sentiment", args.label, args.mode))
    cache_path = os.path.join(out_dir, "api_cache.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                cache[d["key"]] = d
    todo = {}
    for t in texts:
        k = key_of(model, t)
        if k not in cache:
            todo[k] = t
    print("共 %d 行，去重后 %d 条，已缓存 %d，待调用 %d" % (len(texts), len(set(texts)), len(cache), len(todo)))

    with open(cache_path, "a", encoding="utf-8") as fc, \
            cf.ThreadPoolExecutor(max_workers=args.max_workers or int(cfg.get("max_workers", 4))) as ex:
        futs = {ex.submit(call, base, key, model, t, cfg["temperature"]): k for k, t in todo.items()}
        for i, fut in enumerate(cf.as_completed(futs)):
            k = futs[fut]
            answer, usage = fut.result()
            d = {"key": k, "answer": answer, "usage": usage}
            cache[k] = d
            fc.write(json.dumps(d, ensure_ascii=False) + "\n")
            if i % 500 == 0:
                fc.flush()
                print("  已完成 %d / %d" % (i + 1, len(todo)))

    lines, n_unparsed = [], 0
    for t in texts:
        idx = parse(cache[key_of(model, t)]["answer"])
        if idx is None:  # 无法解析的回答记为中性，并计数写进 run_meta
            n_unparsed += 1
            idx = 1
        conf = [0.0, 0.0, 0.0]
        conf[idx] = 1.0
        lines.append("%d %r %r %r" % (idx, conf[0], conf[1], conf[2]))
    hf_utils.write_lines(os.path.join(out_dir, "preds.txt"), lines)
    meta.add(inputs_file=file_info(inputs), n_examples=len(texts), n_unique=len(set(texts)), n_unparsed=n_unparsed,
             out_dir=rel(out_dir))
    meta.save(out_dir)
    print("写出 %s（无法解析 %d 条，按中性处理）" % (rel(os.path.join(out_dir, "preds.txt")), n_unparsed))


if __name__ == "__main__":
    main()
