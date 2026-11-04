# -*- coding: utf-8 -*-
"""【环境 M】HuggingFace 模型加载与批量推理的公共函数。"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from common.paths import CONFIG_DIR, REPO_ROOT


def load_config(name_or_path: str) -> dict:
    path = name_or_path
    if not os.path.exists(path):
        path = os.path.join(CONFIG_DIR, name_or_path if name_or_path.endswith(".yaml") else name_or_path + ".yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = os.path.relpath(path, REPO_ROOT)
    return cfg


def model_entry(cfg: dict, key: str) -> dict:
    if key not in cfg.get("models", {}):
        raise KeyError(f"配置 {cfg['_config_path']} 里没有模型 '{key}'，可选：{list(cfg.get('models', {}))}")
    entry = dict(cfg["models"][key])
    src = entry.get("hf_id") or entry.get("local_path")
    if not src:
        raise ValueError(f"模型 '{key}' 既没有 hf_id 也没有 local_path，请先在 {cfg['_config_path']} 中填写")
    if entry.get("local_path") and not os.path.isabs(entry["local_path"]):
        entry["local_path"] = os.path.join(REPO_ROOT, entry["local_path"])
    entry["source"] = entry.get("local_path") or entry["hf_id"]
    return entry


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def pick_device(device: str = "auto"):
    import torch
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:  # 昇腾 NPU（需要 torch_npu）
        import torch_npu  # noqa: F401
        if torch.npu.is_available():  # type: ignore[attr-defined]
            return torch.device("npu")
    except Exception:
        pass
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_seq_cls(source: str, revision: Optional[str] = None, device: str = "auto"):
    """加载序列分类模型，返回 (tokenizer, model, device, model_info)。"""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(source, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(source, revision=revision)
    dev = pick_device(device)
    model.to(dev).eval()
    info = {
        "source": source,
        "revision_requested": revision,
        "commit_hash": getattr(model.config, "_commit_hash", None),
        "architectures": getattr(model.config, "architectures", None),
        "id2label": getattr(model.config, "id2label", None),
        "num_labels": model.config.num_labels,
        "device": str(dev),
        "dtype": str(next(model.parameters()).dtype),
        "torch_threads": torch.get_num_threads(),
    }
    return tok, model, dev, info


def _batches(n: int, bs: int):
    for i in range(0, n, bs):
        yield i, min(i + bs, n)


def predict_logits(tok, model, device, texts: Sequence, pairs: Optional[Sequence] = None,
                   batch_size: int = 64, max_length: int = 128, desc: str = "predict") -> Tuple[np.ndarray, int]:
    """批量前向，返回 (logits[N, C], 被截断的样本数)。按长度排序分批以提速，输出保持原顺序。"""
    import torch
    from tqdm.auto import tqdm
    n = len(texts)
    enc_len = tok(list(texts), list(pairs) if pairs is not None else None, truncation=False)["input_ids"]
    lengths = np.array([len(x) for x in enc_len])
    n_trunc = int((lengths > max_length).sum())
    order = np.argsort(-lengths, kind="stable")
    out = None
    with torch.inference_mode():
        for s, e in tqdm(list(_batches(n, batch_size)), desc=desc):
            idx = order[s:e]
            a = [texts[i] for i in idx]
            b = [pairs[i] for i in idx] if pairs is not None else None
            enc = tok(a, b, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            logits = model(**enc).logits.float().cpu().numpy()
            if out is None:
                out = np.zeros((n, logits.shape[1]), dtype=np.float64)
            out[idx] = logits
    return out, n_trunc


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def describe_labels(info: Dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in (info.get("id2label") or {}).items())


def write_lines(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")
