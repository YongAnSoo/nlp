# 分工与模型（P22 · CheckList）

选题：P22 CheckList（Ribeiro et al., ACL 2020），限定复现任务「MFT、INV、DIR 行为测试并重跑模型」，资源 C/G。

## 分工

原图：[team_division.png](team_division.png)

| 成员 | 负责内容 | 产出 |
| --- | --- | --- |
| A：情感分析（表 1 + 图 1） | 加载情感测试套件；在 BERT、RoBERTa 的 SST-2 模型上跑测；处理中性阈值；可选加测一个现代模型或现有 API | 复现的表 1，与原文对比 |
| B：QQP（表 2） | 下载或微调 BERT、RoBERTa 的 QQP 模型，先确认它们在标准验证集上准确率正常；跑 QQP 套件 | 复现的表 2，以及"准确率高、失败率也高"的对照 |
| C：SQuAD（表 3） | 用 HF 上的 BERT SQuAD 模型跑阅读理解套件；分析指代、否定等失败案例 | 复现的表 3 和错误案例分析 |
| D：工具与用户研究（表 4）+ 统筹 | 第一周搭好统一环境给大家用；用模板、掩码补全、扰动做演示；组织一次小规模用户研究（如找 6 位同学，分三组，各 30 分钟测同一个模型），复现表 4 的趋势；汇总报告 | 环境与演示 notebook、迷你用户研究、最终报告 / PPT 的整合 |

## 使用的模型

| 对应原文 | 我们用的模型 | 是否需要训练 | 负责人 |
| --- | --- | --- | --- |
| #4 | `textattack/bert-base-uncased-SST-2`（HF 上已确认存在） | 否 | A |
| #5 | `textattack/roberta-base-SST-2`（已确认存在） | 否 | A |
| #6 | `textattack/bert-base-uncased-QQP`（已确认存在） | 否 | B |
| #7 | RoBERTa-base 的 QQP 模型：textattack 没有发布这个版本（有 CoLA、MRPC 版，没有 QQP） | 要么在 HF 上找其他人上传的，要么自己微调；可能是全组唯一必须训练的模型 | B |
| #8 | `bert-large-uncased-whole-word-masking-finetuned-squad` | 否 | C |

## 材料位置

| 内容 | 路径 |
| --- | --- |
| A 的 notebook（Kaggle） | `code/sentiment/membera-checklist.ipynb` |
| A 的结果包（不含 `reference/` 下 4 个官方文件的副本） | `results/sentiment/member_a_results/` |
| A 的结果报告 | `report/member_a_report.md` |
| D 的统一环境 checklist39（原样） | `envs/requirements-checklist39.txt` |

`reference/` 里的 `sentiment_suite.pkl`、`tests_n500` 和作者的 BERT / RoBERTa 预测与官方 release 逐字节一致，sha256 记在 `results/sentiment/member_a_results/experiment_config.json`。可用 `python code/setup/download_release_data.py` 重新取得。
