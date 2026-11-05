
import sys
import json
import csv
import numpy as np
from pathlib import Path

sys.path.insert(0, "/kaggle/working/checklist")
from checklist.test_suite import TestSuite

out = Path("/kaggle/working/member_a_results")
ref = out / "reference"

selected = [
    ("中性短句",
     "neutral words in context",
     "neutral"),
    ("否定中性表达",
     "simple negations: not neutral is still neutral",
     "neutral"),
    ("句末否定负面表达",
     "simple negations: I thought x was negative, but it was not (should be neutral or positive)",
     "neutral or positive"),
    ("插入中性内容的否定",
     "Hard: Negation of positive with neutral stuff in the middle (should be negative)",
     "negative"),
    ("作者观点优先",
     "my opinion is what matters",
     "see suite label"),
]

suites = {}
for model in ["bert", "roberta"]:
    suite = TestSuite.from_file(str(ref / "sentiment_suite.pkl"))
    suite.run_from_file(
        str(out / f"{model}_own_predictions.txt"),
        overwrite=True,
    )
    suites[model] = suite

label_names = {0: "negative", 1: "neutral", 2: "positive"}

def data_index(test, result_index):
    if test.run_idxs is None:
        return int(result_index)
    return int(test.run_idxs[result_index])

def expected_label(test, idx, fallback):
    labels = test.labels
    if labels is None:
        return fallback
    if isinstance(labels, (list, tuple, np.ndarray)):
        value = labels[idx]
    else:
        value = labels
    if np.isscalar(value):
        return label_names.get(int(value), str(value))
    return str(value)

records = []

for title, name, fallback in selected:
    bert = suites["bert"].tests[name]
    roberta = suites["roberta"].tests[name]

    b_ids = (
        list(range(len(bert.data))) if bert.run_idxs is None
        else [int(x) for x in bert.run_idxs]
    )
    r_ids = (
        list(range(len(roberta.data))) if roberta.run_idxs is None
        else [int(x) for x in roberta.run_idxs]
    )
    assert b_ids == r_ids, "两个模型的抽样索引不一致"

    b_fails = set(int(x) for x in bert.fail_idxs())
    r_fails = set(int(x) for x in roberta.fail_idxs())

    # 每类最多取两个案例：
    # 共同失败的第一个，以及只有一个模型失败的第一个。
    groups = [
        ("both_fail", sorted(b_fails & r_fails)),
        ("one_model_fails", sorted(b_fails ^ r_fails)),
    ]

    for category, candidates in groups:
        if not candidates:
            continue

        result_idx = candidates[0]
        idx = data_index(bert, result_idx)
        text = bert.data[idx]
        assert isinstance(text, str), "本单元仅提取单句MFT案例"
        assert text == roberta.data[idx]

        record = {
            "test_zh": title,
            "official_test_name": name,
            "selection": category,
            "suite_data_index": idx,
            "text": text,
            "expected": expected_label(bert, idx, fallback),
        }

        for model, test, fails in [
            ("bert", bert, b_fails),
            ("roberta", roberta, r_fails),
        ]:
            prediction = int(test.results.preds[idx])
            confs = np.asarray(test.results.confs[idx], dtype=float)
            assert confs.shape == (3,)
            record[f"{model}_prediction"] = label_names[prediction]
            record[f"{model}_failed"] = result_idx in fails
            record[f"{model}_confs"] = confs.tolist()

        records.append(record)

        print("\n测试:", title)
        print("案例选择:", category)
        print("输入:", text)
        print("期望:", record["expected"])
        for model in ["bert", "roberta"]:
            state = "FAIL" if record[f"{model}_failed"] else "PASS"
            probs = ", ".join(
                f"{x:.4f}" for x in record[f"{model}_confs"]
            )
            print(
                f"{model}: {record[f'{model}_prediction']} | {state}"
                f" | [负面, 中性, 正面]=[{probs}]"
            )

assert records, "没有提取到案例"

(out / "failure_cases.json").write_text(
    json.dumps(records, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

with (out / "failure_cases.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)

print("\nFAILURE_CASES_OK")
