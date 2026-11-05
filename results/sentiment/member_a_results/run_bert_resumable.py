
import os
os.environ["USE_TF"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import json
import hashlib
import shutil
import numpy as np
import torch
import transformers
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

out = Path("/kaggle/working/member_a_results")
config = json.loads((out / "experiment_config.json").read_text())
spec = config["models"]["bert"]
source = out / "reference/tests_n500"

with source.open(encoding="utf-8") as f:
    texts = [line.rstrip("\n") for line in f]

source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
assert source_hash == config["sha256"]["tests_n500"]
assert len(texts) == 87470
assert torch.cuda.is_available(), "GPU 未连接"

# 用于验证断点属于同一次实验
identity = {
    "model_id": spec["id"],
    "revision": spec["revision"],
    "input_sha256": source_hash,
    "max_length": 512,
    "batch_size": 32,
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
}

checkpoint = out / "bert_progress.npz"
probabilities = []
truncated = 0

if checkpoint.exists():
    with np.load(checkpoint, allow_pickle=False) as saved:
        assert json.loads(saved["identity"].item()) == identity, \
            "断点配置不同，请先核对，不能混用结果"
        probabilities = saved["positive_probs"].tolist()
        truncated = int(saved["truncated"].item())
    assert len(probabilities) <= len(texts)
    assert np.isfinite(probabilities).all()
    assert all(0 <= p <= 1 for p in probabilities)

print("GPU:", torch.cuda.get_device_name(0), flush=True)
print(f"从 {len(probabilities)}/{len(texts)} 条继续", flush=True)

def save_progress():
    temp = out / "bert_progress.tmp.npz"
    np.savez(
        temp,
        positive_probs=np.asarray(probabilities, dtype=np.float64),
        truncated=np.asarray(truncated),
        identity=np.asarray(json.dumps(identity, sort_keys=True)),
    )
    os.replace(temp, checkpoint)

if len(probabilities) < len(texts):
    tokenizer = AutoTokenizer.from_pretrained(
        spec["id"], revision=spec["revision"]
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        spec["id"], revision=spec["revision"]
    ).to("cuda:0").eval()

    for batch_no, start in enumerate(
        range(len(probabilities), len(texts), 32), start=1
    ):
        batch_texts = texts[start:start + 32]
        ids = tokenizer(
            batch_texts, truncation=False, padding=False
        )["input_ids"]
        batch_truncated = sum(len(x) > 512 for x in ids)

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to("cuda:0")

        with torch.inference_mode():
            p = model(**inputs).logits.softmax(dim=-1)[:, 1]
        probabilities.extend(p.cpu().tolist())
        truncated += batch_truncated

        if batch_no % 100 == 0 or len(probabilities) == len(texts):
            save_progress()
            print(
                f"已保存进度：{len(probabilities)}/{len(texts)}",
                flush=True,
            )

# 转为官方评分所需的三类置信度
p = np.asarray(probabilities, dtype=np.float64)
assert len(p) == len(texts)
assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()

lo = 0.5 - (1 / 3) / 2
hi = 0.5 + (1 / 3) / 2
q = np.zeros((len(p), 3), dtype=np.float64)

outer = (p <= lo) | (p >= hi)
q[outer, 0] = 1 - p[outer]
q[outer, 2] = p[outer]

middle = ~outer
q[middle, 1] = 1 - 3 * np.abs(p[middle] - 0.5)
left = middle & (p < 0.5)
right = middle & (p >= 0.5)
q[left, 0] = 1 - q[left, 1]
q[right, 2] = 1 - q[right, 1]

assert np.allclose(q.sum(axis=1), 1)
assert ((q >= 0) & (q <= 1)).all()
preds = q.argmax(axis=1)

np.save(out / "bert_own_positive_probs.npy", p)

pred_path = out / "bert_own_predictions.txt"
temp_path = out / "bert_own_predictions.tmp"
with temp_path.open("w") as f:
    for label, row in zip(preds, q):
        f.write(
            f"{label} {row[0]:.12f} {row[1]:.12f} {row[2]:.12f}\n"
        )
os.replace(temp_path, pred_path)

metadata = {
    **identity,
    "source": "own_model_inference",
    "n_texts": len(texts),
    "truncated_examples": truncated,
    "neutral_rule": "lo < p_positive < hi",
    "lo": lo,
    "hi": hi,
    "exact_boundary_count": int(((p == lo) | (p == hi)).sum()),
    "class_counts_negative_neutral_positive":
        np.bincount(preds, minlength=3).tolist(),
}
(out / "bert_own_metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)

# ZIP 放在结果文件夹外，避免把备份自身再次打包
archive = shutil.make_archive(
    "/kaggle/working/member_a_bert_backup",
    "zip",
    "/kaggle/working",
    "member_a_results",
)

# 检查备份完整性
import zipfile
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert "member_a_results/bert_own_predictions.txt" in z.namelist()

print("\nBERT_INFERENCE_OK", flush=True)
print("截断样本数:", truncated, flush=True)
print("负面/中性/正面数量:",
      metadata["class_counts_negative_neutral_positive"], flush=True)
print("备份文件:", archive, flush=True)
print("请现在手动下载备份，下载完成后再继续！", flush=True)
