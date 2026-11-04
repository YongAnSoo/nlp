# -*- coding: utf-8 -*-
"""SQuAD v1.1 官方评测的 EM / F1（按官方 evaluate-v1.1.py 的归一化规则实现）。"""
import collections
import re
import string


def normalize_answer(s):
    """小写、去标点、去冠词 a/an/the、合并空白。"""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(fn, prediction, ground_truths):
    return max(fn(prediction, gt) for gt in ground_truths)


def evaluate(predictions, references):
    """predictions: [str]；references: [[str, ...]]（每题可有多个参考答案）。返回百分制 EM/F1。"""
    assert len(predictions) == len(references)
    em = f1 = 0.0
    for p, refs in zip(predictions, references):
        em += metric_max_over_ground_truths(exact_match_score, p, refs)
        f1 += metric_max_over_ground_truths(f1_score, p, refs)
    n = len(predictions)
    return {"exact_match": 100.0 * em / n, "f1": 100.0 * f1 / n, "n": n}
