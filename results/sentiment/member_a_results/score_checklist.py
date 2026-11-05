
import sys
import csv
import json
import hashlib
import subprocess
import numpy as np
from pathlib import Path

# 脚本保存在结果目录，显式指定官方源码位置
sys.path.insert(0, "/kaggle/working/checklist")
from checklist.test_suite import TestSuite

out = Path("/kaggle/working/member_a_results")
ref = out / "reference"

with (ref / "tests_n500").open(encoding="utf-8") as f:
    texts = [line.rstrip("\n") for line in f]
input_hash = hashlib.sha256((ref / "tests_n500").read_bytes()).hexdigest()

# 表1中的17项测试及论文失败率
table1 = [
    ("中性短句", "neutral words in context", 94.6, 81.8),
    ("情感短句", "Sentiment-laden words in context", 0.0, 0.2),
    ("替换中性词", "change neutral words with BERT", 10.2, 10.2),
    ("添加正面短语", "add positive phrases", 0.2, 10.2),
    ("添加负面短语", "add negative phrases", 0.0, 13.2),
    ("添加网址和账号名", "add random urls and handles", 11.4, 7.4),
    ("相邻字符拼写错误", "typos", 5.2, 3.8),
    ("替换地名", "change locations", 7.6, 6.4),
    ("替换人名", "change names", 6.6, 2.4),
    ("过去与现在的情感变化", "used to, but now", 18.8, 11.0),
    ("否定负面表达", "simple negations: not negative", 13.2, 2.6),
    ("否定中性表达",
     "simple negations: not neutral is still neutral", 98.4, 95.4),
    ("句末否定负面表达",
     "simple negations: I thought x was negative, but it was not (should be neutral or positive)",
     84.8, 7.2),
    ("插入中性内容的否定",
     "Hard: Negation of positive with neutral stuff in the middle (should be negative)",
     74.0, 30.2),
    ("作者观点优先", "my opinion is what matters", 38.8, 30.0),
    ("问答：Yes", "Q & A: yes", 3.6, 3.0),
    ("问答：No", "Q & A: no", 55.4, 54.8),
]

all_rows = []
own_rates = {}

for model in ["bert", "roberta"]:
    suite = TestSuite.from_file(str(ref / "sentiment_suite.pkl"))

    # 保持官方抽样与文本顺序，不重新抽样
    assert suite.get_raw_examples(new_sample=False) == texts, \
        "输入文本与套件顺序不一致"

    # 先验证官方预测能够重现表1
    suite.run_from_file(
        str(ref / f"official_{model}_predictions.txt"),
        overwrite=True,
    )
    for title, name, paper_b, paper_r in table1:
        paper = paper_b if model == "bert" else paper_r
        rate = suite.tests[name].get_stats()["fail_rate"]
        assert round(rate, 1) == paper, \
            f"官方回放与论文不一致：{model}, {name}, {rate}"

    # 检查自己的预测文件
    metadata = json.loads(
        (out / f"{model}_own_metadata.json").read_text()
    )
    assert metadata["input_sha256"] == input_hash
    assert metadata["n_texts"] == len(texts)

    pred_path = out / f"{model}_own_predictions.txt"
    values = np.loadtxt(str(pred_path))
    assert values.shape == (len(texts), 4)
    assert np.isfinite(values).all()
    assert np.isin(values[:, 0], [0, 1, 2]).all()
    assert ((values[:, 1:] >= 0) & (values[:, 1:] <= 1)).all()
    assert np.allclose(values[:, 1:].sum(axis=1), 1)
    assert np.array_equal(values[:, 0], values[:, 1:].argmax(axis=1))

    suite.run_from_file(str(pred_path), overwrite=True)
    own_rates[model] = {}

    for name, test in suite.tests.items():
        stats = test.get_stats()
        n_run = stats.get("testcases_run", stats["testcases"])
        n_valid = stats.get("after_filtering", n_run)
        rate = stats.get("fail_rate")
        own_rates[model][name] = rate

        all_rows.append({
            "source": "own_model_inference",
            "model": model,
            "test": name,
            "capability": suite.info[name]["capability"],
            "type": suite.info[name]["type"],
            "n_run": int(n_run),
            "n_valid": int(n_valid),
            "n_failed": int(stats.get("fails", 0)),
            "failure_rate_pct": "" if rate is None else rate,
        })

    print(model, "官方17项回放校验通过；自己的38项评分完成",
          flush=True)

def save_csv(filename, rows):
    with (out / filename).open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

save_csv("own_all_38_tests.csv", all_rows)

comparison = []
print("\n失败率单位：%；差值为实测减原文，单位为百分点")
print("测试\tBERT原文\tBERT实测\t差值\tRoBERTa原文\tRoBERTa实测\t差值")

for title, name, paper_b, paper_r in table1:
    own_b = own_rates["bert"][name]
    own_r = own_rates["roberta"][name]
    assert own_b is not None and own_r is not None

    comparison.append({
        "test_zh": title,
        "official_test_name": name,
        "bert_paper_pct": paper_b,
        "bert_own_pct": own_b,
        "bert_delta_pp": own_b - paper_b,
        "roberta_paper_pct": paper_r,
        "roberta_own_pct": own_r,
        "roberta_delta_pp": own_r - paper_r,
    })
    print(
        f"{title}\t{paper_b:.1f}\t{own_b:.1f}\t{own_b-paper_b:+.1f}"
        f"\t{paper_r:.1f}\t{own_r:.1f}\t{own_r-paper_r:+.1f}"
    )

save_csv("table1_reproduction_comparison.csv", comparison)

freeze = subprocess.check_output(
    [sys.executable, "-m", "pip", "freeze"],
    universal_newlines=True,
)
(out / "scoring_requirements.txt").write_text(freeze)
print("\nCHECKLIST_SCORING_OK")
