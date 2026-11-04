# -*- coding: utf-8 -*-
"""CheckList 套件的加载、输入导出、对齐校验、打分与逐条结果导出。

本模块在「环境 E（Python 3.7）」里运行（因为官方 *_suite.pkl 只能在 Py≤3.7 反序列化），
所以写法保持 Python 3.6 兼容：不用海象运算符、不用 dataclass、不用 str.removeprefix 等。

术语（与原文一致）：
  test（测试）      = 套件里的一项，如 "Sentiment-laden words in context"，类型为 MFT / INV / DIR
  test case（用例） = 测试里的一条，可能包含 1 个或多个 example（INV/DIR 为原句+扰动句）
  example（样本）   = 送进模型的一条输入；预测文件每行对应一个 example
  fail rate         = 失败用例数 / 未被过滤的用例数 × 100（checklist 的 AbstractTest.get_stats）
"""
import collections
import csv
import gzip
import io
import json
import math
import os
import sys

import numpy as np

from common.paths import RELEASE_DIR, GENERATED_DIR, ensure_dir, rel

# ---------------------------------------------------------------------------
# 任务定义
# ---------------------------------------------------------------------------
TASKS = collections.OrderedDict([
    ("sentiment", {
        "suite": "sentiment/sentiment_suite.pkl",
        "released_inputs": "sentiment/tests_n500",
        "full_inputs": "sentiment/full_inputs.txt",
        # 每行：预测(0负/1中/2正) + 三类概率(负 中 正)
        "pred_format": "pred_and_softmax",
        "released_models": ["bert", "roberta", "google", "amazon", "microsoft"],
        "paper_table": "Table 1",
    }),
    ("qqp", {
        "suite": "qqp/qqp_suite.pkl",
        "released_inputs": "qqp/tests_n500",
        "full_inputs": "qqp/full_inputs.tsv",
        # 每行：一个数，P(duplicate)
        "pred_format": "binary_conf",
        "released_models": ["bert", "roberta"],
        "paper_table": "Table 2",
    }),
    ("squad", {
        "suite": "squad/squad_suite.pkl",
        "released_inputs": "squad/squad.jsonl",
        "full_inputs": "squad/full_inputs.jsonl",
        # 每行：预测的答案片段（字符串）
        "pred_format": "pred_only",
        "released_models": ["bert"],
        "paper_table": "Table 3",
    }),
])
MODES = ("released", "full")

# 原文表格的能力（capability）顺序，同 checklist.test_suite.summary
CAPABILITY_ORDER = ["Vocabulary", "Taxonomy", "Robustness", "NER", "Fairness", "Temporal",
                    "Negation", "Coref", "SRL", "Logic"]


def suite_path(task):
    return os.path.join(RELEASE_DIR, TASKS[task]["suite"])


def input_path(task, mode):
    """predictor 读取的输入文件。released = 官方文件；full = 由 export_inputs.py 导出。"""
    if mode == "released":
        return os.path.join(RELEASE_DIR, TASKS[task]["released_inputs"])
    if mode == "full":
        return os.path.join(GENERATED_DIR, TASKS[task]["full_inputs"])
    raise ValueError("mode 必须是 %s" % (MODES,))


def released_pred_path(task, model):
    return os.path.join(RELEASE_DIR, task, "predictions", model)


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------
PY_VERSION_HELP = (
    "官方 *_suite.pkl 里用 dill 存了 Python≤3.7 的字节码（期望函数），只能在 Python 3.7（或 3.6）"
    "里加载。请切换到环境 E：conda activate checklist-eval37（见 README「两套环境」）。")


def load_suite(task=None, path=None, strict_python=True):
    """加载 TestSuite。

    strict_python=True 时，在 Python≥3.8 上直接报错并给出说明（官方 pkl 必然加载失败）。
    自己在新版 Python 里生成并保存的 suite（如单元测试里的合成套件）可传 strict_python=False。
    """
    from checklist.test_suite import TestSuite
    p = path or suite_path(task)
    if not os.path.exists(p):
        raise IOError("找不到 suite 文件 %s。先运行 python code/setup/download_release_data.py" % p)
    if strict_python and sys.version_info >= (3, 8):
        raise RuntimeError("当前 Python %s。%s" % (sys.version.split()[0], PY_VERSION_HELP))
    try:
        return TestSuite.from_file(p)
    except TypeError as e:
        raise RuntimeError("加载 %s 失败：%s\n%s" % (p, e, PY_VERSION_HELP))


# ---------------------------------------------------------------------------
# 样本格式（与官方 release 文件逐字一致）
# ---------------------------------------------------------------------------
def _identity(x):
    return x


def normalize_example(task, x):
    """把 suite 里的原始 example 统一成：sentiment -> str；qqp -> (q1, q2)；squad -> (passage, question)。"""
    if task == "sentiment":
        return str(x).replace("\n", " ")
    if task == "qqp":
        if isinstance(x, (list, tuple)) and len(x) == 2:
            return (str(x[0]), str(x[1]))
        raise ValueError("QQP example 应为 (q1, q2)，实际为 %r" % (x,))
    if task == "squad":
        if isinstance(x, dict):
            passage = x.get("passage", x.get("context"))
            return (passage, x["question"])
        if isinstance(x, (list, tuple)) and len(x) == 2:
            return (x[0], x[1])
        raise ValueError("SQuAD example 应为 (passage, question) 或 dict，实际为 %r" % (x,))
    raise ValueError("未知任务 %s" % task)


def format_line(task, ex):
    """与 checklist TestSuite.to_raw_file 的输出格式一致（不含 qqp 的 id 前缀）。"""
    if task == "sentiment":
        return ex
    if task == "qqp":
        return "\t".join(ex).replace("\n", " ")
    if task == "squad":
        return json.dumps({"passage": ex[0], "question": ex[1]})
    raise ValueError(task)


def read_input_file(task, path):
    """读取输入文件，返回 normalize 后的样本列表（predictor 与对齐校验共用）。"""
    with io.open(path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if task == "sentiment":
        return lines
    if task == "qqp":
        if not lines or lines[0].strip() != "id\tquestion1\tquestion2":
            raise ValueError("QQP 输入文件应以表头 id\\tquestion1\\tquestion2 开始：%s" % path)
        out = []
        for i, line in enumerate(lines[1:]):
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError("%s 第 %d 行不是 3 列：%r" % (path, i + 2, line))
            if int(parts[0]) != i:
                raise ValueError("%s 第 %d 行 id=%s，期望 %d" % (path, i + 2, parts[0], i))
            out.append((parts[1], parts[2]))
        return out
    if task == "squad":
        out = []
        for line in lines:
            d = json.loads(line)
            out.append((d["passage"], d["question"]))
        return out
    raise ValueError(task)


def write_input_file(task, examples, path):
    ensure_dir(os.path.dirname(path))
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        if task == "qqp":
            f.write(u"id\tquestion1\tquestion2\n")
            for i, ex in enumerate(examples):
                f.write(u"%d\t%s\n" % (i, format_line(task, ex)))
        else:
            for ex in examples:
                f.write(format_line(task, ex) + u"\n")


def example_list(suite, task, mode):
    """返回该模式下、按预测文件行序排列的全部样本。

    released：沿用 pkl 里保存的抽样（每个测试至多 500 个用例，即官方 tests_n500 / squad.jsonl），
              不修改 suite 状态；
    full    ：重新展开 *全部* 用例（n=None，无抽样），会改写 suite.test_ranges / test.run_idxs /
              test.result_indexes，使后续打分与 full 输入文件逐行对齐。
    """
    if mode == "released":
        if not suite.test_ranges:
            raise RuntimeError("suite 里没有保存抽样信息（test_ranges 为空），无法使用 released 模式")
        raw = suite.get_raw_example_list(file_format=None, format_fn=_identity)
    elif mode == "full":
        raw = suite.create_raw_example_list(file_format=None, format_fn=_identity, n=None, seed=None)
    else:
        raise ValueError("mode 必须是 %s" % (MODES,))
    return [normalize_example(task, x) for x in raw]


def check_alignment(task, suite_examples, file_examples, path):
    """逐行比较 suite 展开的样本与输入文件，保证预测文件第 i 行就是第 i 个样本。"""
    if len(suite_examples) != len(file_examples):
        raise RuntimeError("行数不一致：suite 展开 %d 条，%s 有 %d 条"
                           % (len(suite_examples), rel(path), len(file_examples)))
    for i, (a, b) in enumerate(zip(suite_examples, file_examples)):
        if task == "qqp":
            same = format_line(task, a) == format_line(task, b)
        else:
            same = (a == b) or (tuple(a) == tuple(b) if task == "squad" else False)
        if not same:
            raise RuntimeError("第 %d 行不一致：\n  suite: %r\n  file : %r" % (i, a, b))
    return True


def tests_index(suite):
    """每个测试在输入文件中的行范围，写成 json 方便在环境 M 里查看。"""
    rows = []
    for name, test in suite.tests.items():
        info = suite.info[name]
        start, end = suite.test_ranges[name]
        n_run = len(test.run_idxs) if test.run_idxs is not None else len(test.data)
        rows.append(collections.OrderedDict([
            ("test_name", name), ("capability", info.get("capability") or ""),
            ("type", info.get("type") or ""), ("description", info.get("description") or ""),
            ("line_start", int(start)), ("line_end", int(end)),
            ("n_testcases", len(test.data)), ("n_testcases_in_file", int(n_run)),
        ]))
    return rows


# ---------------------------------------------------------------------------
# 打分
# ---------------------------------------------------------------------------
def read_predictions(task, pred_path, pred_format=None):
    from checklist.abstract_test import read_pred_file
    fmt = pred_format or TASKS[task]["pred_format"]
    return read_pred_file(pred_path, file_format=fmt)


def run_suite(suite, task, mode, preds, confs, inputs_path=None):
    """在 suite 上用 (preds, confs) 打分。会先做行数与（可选）输入文件对齐检查。"""
    examples = example_list(suite, task, mode)
    if inputs_path is not None:
        check_alignment(task, examples, read_input_file(task, inputs_path), inputs_path)
    if len(preds) != len(examples):
        raise RuntimeError("预测 %d 行，但 %s 模式下 suite 共有 %d 个样本" % (len(preds), mode, len(examples)))
    suite.run_from_preds_confs(preds, confs, overwrite=True)
    return examples


def _cap_key(cap):
    return CAPABILITY_ORDER.index(cap) if cap in CAPABILITY_ORDER else len(CAPABILITY_ORDER)


def per_test_rows(suite, task, model_label, mode):
    """每个测试一行，数值直接取自 checklist 的 get_stats()（与 suite.summary() 打印的一致）。"""
    rows = []
    for order, (name, test) in enumerate(suite.tests.items()):
        info = suite.info[name]
        st = test.get_stats()
        n_run = st.get("testcases_run", st["testcases"])
        rows.append(collections.OrderedDict([
            ("task", task), ("model", model_label), ("mode", mode),
            ("capability", info.get("capability") or ""), ("type", info.get("type") or ""),
            ("test_name", name), ("description", info.get("description") or ""),
            ("n_testcases", st["testcases"]), ("n_run", n_run),
            ("n_after_filtering", st.get("after_filtering", n_run)),
            ("n_fail", st.get("fails", "")),
            ("fail_rate", st["fail_rate"] if "fail_rate" in st else ""),
            ("suite_order", order),
        ]))
    rows.sort(key=lambda r: (_cap_key(r["capability"]), r["suite_order"]))
    return rows


def to_jsonable(x):
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return [to_jsonable(v) for v in x.tolist()]
    if isinstance(x, dict):
        return collections.OrderedDict((str(k), to_jsonable(v)) for k, v in x.items())
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return str(x)


def iter_testcase_records(suite, task, model_label, mode):
    """逐用例（test case）输出：ID、输入行号、样本、标签、预测、置信度、期望结果、是否通过。"""
    for name, test in suite.tests.items():
        info = suite.info[name]
        # 用 suite.run() 直接跑（不经过输入文件）时没有 test_ranges，此时 input_lines 为空
        rng = suite.test_ranges.get(name) if suite.test_ranges else None
        lines_by_case = collections.defaultdict(list)
        if rng is not None:
            start = int(rng[0])
            for k, ti in enumerate(test.result_indexes):
                lines_by_case[int(ti)].append(start + k)
        run_idxs = test.run_idxs if test.run_idxs is not None else range(len(test.data))
        passed = test.results.passed
        expect = test.results.expect_results
        for f, d_idx in enumerate(run_idxs):
            d_idx = int(d_idx)
            label, meta = test._label_meta(d_idx)
            p = passed[f]
            yield collections.OrderedDict([
                ("id", "%s::%s::%d" % (task, name, d_idx)),
                ("task", task), ("model", model_label), ("mode", mode),
                ("test_name", name), ("capability", info.get("capability") or ""),
                ("type", info.get("type") or ""), ("testcase_idx", d_idx),
                ("input_lines", lines_by_case.get(d_idx, [])),
                ("examples", to_jsonable(test.data[d_idx])),
                ("label", to_jsonable(label)), ("meta", to_jsonable(meta)),
                ("preds", to_jsonable(test.results.preds[d_idx])),
                ("confs", to_jsonable(test.results.confs[d_idx])),
                ("expect_results", to_jsonable(expect[f])),
                # None = 该用例被期望函数过滤（不计入失败率分母）
                ("passed", None if p is None else bool(p)),
            ])


# ---------------------------------------------------------------------------
# 写结果
# ---------------------------------------------------------------------------
PER_TEST_FIELDS = ["task", "model", "mode", "capability", "type", "test_name", "description",
                   "n_testcases", "n_run", "n_after_filtering", "n_fail", "fail_rate", "suite_order"]


def write_per_test_csv(rows, path):
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_TEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_jsonl_gz(records, path):
    n = 0
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def fmt_rate(v):
    return "" if v == "" or v is None else "%.1f" % float(v)


def write_summary_md(rows, path, title):
    lines = ["# %s" % title, "",
             "失败率 = 失败用例数 / 过滤后用例数 × 100（checklist `get_stats()`）。", "",
             "| Capability | Type | Test | #cases(run) | #after filter | #fail | Fail % |",
             "|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            r["capability"], r["type"], r["test_name"].replace("|", "\\|"), r["n_run"],
            r["n_after_filtering"], r["n_fail"], fmt_rate(r["fail_rate"])))
    with io.open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def evaluate_to_dir(suite, task, mode, preds, confs, model_label, out_dir, inputs_path=None):
    """打分并写出 per_test.csv / summary.md / per_example.jsonl.gz，返回 (rows, n_records)。"""
    ensure_dir(out_dir)
    run_suite(suite, task, mode, preds, confs, inputs_path=inputs_path)
    rows = per_test_rows(suite, task, model_label, mode)
    write_per_test_csv(rows, os.path.join(out_dir, "per_test.csv"))
    write_summary_md(rows, os.path.join(out_dir, "summary.md"),
                     "%s · %s · %s（对应原文 %s）" % (task, model_label, mode, TASKS[task]["paper_table"])
                     if task in TASKS else "%s · %s · %s" % (task, model_label, mode))
    n = write_jsonl_gz(iter_testcase_records(suite, task, model_label, mode),
                       os.path.join(out_dir, "per_example.jsonl.gz"))
    return rows, n
