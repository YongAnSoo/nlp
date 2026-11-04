# P22 · CheckList 部分复现（NLP 项目实践 2026）

> ⚠️ **仓库仍在搭建中（WIP）**。已完成：环境说明、数据下载、CheckList 打分与逐条结果导出、A/B/C 的推理与验证集脚本、QQP 微调脚本。
> 待补：D 的工具演示 notebook 与用户研究材料、单元测试、完整 README、报告模板、`contributions.md`。

论文：Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, Sameer Singh. *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList*. ACL 2020.
代码：https://github.com/marcotcr/checklist（PyPI `checklist==0.0.11`，release_data.tar.gz sha256 `061524b3…fd202`）
限定复现任务：MFT、INV、DIR 行为测试并重跑模型。

## 分工与模型

| 成员 | 内容 | 模型（`configs/*.yaml` 里的键） |
|---|---|---|
| A | 情感分析（表 1 + 图 1） | `bert-textattack` = textattack/bert-base-uncased-SST-2；`roberta-textattack` = textattack/roberta-base-SST-2 |
| B | QQP（表 2） | `bert-textattack` = textattack/bert-base-uncased-QQP；`roberta-finetuned` = 自己微调的 roberta-base（3 个种子） |
| C | SQuAD（表 3） | `bert-large-wwm` = google-bert/bert-large-uncased-whole-word-masking-finetuned-squad |
| D | 工具与用户研究（表 4）+ 统筹 | 环境、演示 notebook、迷你用户研究 |

## 两套环境（重要）

官方 `release_data/*/*_suite.pkl` 里用 dill 存了 **Python ≤3.7 的字节码**（期望函数）。静态检查（未执行）显示，CodeType 参数是 3.7 及以前的布局，SQuAD 套件更像 3.6 编译的。**Python ≥3.8 无法加载**。因此：

| 环境 | Python | 用途 | 安装 |
|---|---|---|---|
| **E（eval）** | 3.7（不行再试 3.6） | 加载套件、导出输入、给预测文件打分 | `conda env create -f envs/environment-eval-py37.yml` → `conda activate checklist-eval37` → `bash code/setup/install_checklist_eval.sh` |
| **M（modern）** | 3.10 / 3.11 | 跑模型、微调、工具演示、用户研究 | `python3.11 -m venv .venv && source .venv/bin/activate` → `bash code/setup/install_env_modern.sh` |

国内下载 HF 模型较慢时可设：`export HF_ENDPOINT=https://hf-mirror.com`

## 流程（以 A 为例，B/C 同理）

```bash
# 0) 下载官方数据（任一环境）
python code/setup/download_release_data.py
# 1) 【环境 E】环境校验：用作者公开的预测复算，数值应与论文表 1/2/3 一致
python code/common/reproduce_released.py --task all
# 2) 【环境 E】导出全部用例（full 模式，不抽样）并校验 released 输入
python code/common/export_inputs.py --task all --mode both
# 3) 【环境 M】验证集准确率（基线指标）+ 套件推理
python code/sentiment/validate_sst2.py --model bert-textattack
python code/sentiment/predict_sentiment.py --model bert-textattack --mode released
python code/sentiment/predict_sentiment.py --model bert-textattack --mode full
# 4) 【环境 E】打分
python code/common/evaluate_suite.py --task sentiment --mode released \
    --pred results/sentiment/bert-textattack/released/preds.txt --label bert-textattack
# 5) 与作者结果并排、导出错误案例
python code/common/make_table.py \
    --run "bert(released)=results/released_reference/sentiment/bert/per_test.csv" \
    --run "bert-textattack=results/sentiment/bert-textattack/released/eval/per_test.csv" \
    --out results/sentiment/table1_compare
python code/common/export_failures.py \
    --per-example results/sentiment/bert-textattack/released/eval/per_example.jsonl.gz
```

- **released 模式**：作者的抽样（每个测试最多 500 个用例），能和论文数值直接对照。
- **full 模式**：套件里的全部用例。
- 两种都跑，报告里分开写。

## 成员 A 进度（2026-11-04 核对）

A 在 Kaggle（2×T4）上完成了 BERT / RoBERTa 在 `tests_n500` 全部 87,470 条上的推理与 38 项打分（打分环境 Python 3.7.12）。
用作者公开预测复算表 1 的 17 项 × 2 个模型，与论文一致（脚本内断言通过）。
核对结果：
- 参考文件与官方 release 逐字节一致；
- A 的预测与本仓库 `code/sentiment/neutral.py` 重算结果 100% 一致（概率差 ≤5e-13，来自 12 位小数舍入）。

有一个值得写进报告的现象：落在中性带（P(pos) ∈ (1/3, 2/3)）的样本数，A 的模型明显多于作者的模型。

| 模型 | 作者公开预测 | A 的预测（textattack） |
|---|---:|---:|
| BERT | 2,228 | 5,871 |
| RoBERTa | 5,489 | 7,999 |

用中性阈值消融检验这对失败率差异的影响。不用重跑模型，两步都在环境 E 里做：

```bash
python code/sentiment/neutral_ablation.py make --p-pos <A的>/bert_own_positive_probs.npy \
    --label bert-textattack --out results/sentiment/bert-textattack/released/neutral_ablation
python code/sentiment/neutral_ablation.py score --dir results/sentiment/bert-textattack/released/neutral_ablation \
    --suite <A的>/reference/sentiment_suite.pkl --inputs <A的>/reference/tests_n500
```

A 还需补：SST-2 验证集准确率（基线）、逐用例结果 `per_example.jsonl.gz`（用 `evaluate_suite.py` 读 A 的预测文件即可）、INV/DIR 类案例及「错误类型 / 解释」两列。

B 的 RoBERTa-QQP 需要自己微调（全部训练集，3 个种子）：

```bash
python code/common/finetune_glue.py --task qqp --model roberta-base --seed 42   # 再跑 43、44
```
