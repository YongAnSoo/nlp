# -*- coding: utf-8 -*-
"""二分类情感模型 -> 三分类（负/中/正）的「中性阈值」处理。保持 Python 3.6 兼容。

CheckList 的情感测试套件含中性用例（如 "I am an American."），但 SST-2 模型只有正/负两类。
作者把 P(positive) 落在 0.5 附近一个宽度为 margin 的区间内的预测视为「中性」。

从官方公开预测（release_data/sentiment/predictions/{bert,roberta}）可以反推出作者用的规则：
  * 预测为正的行 P(pos) 最小为 0.6667，预测为负的行 P(pos) 最大为 0.3333   → margin = 1/3
  * 中性行的格式为 (0, p_neu, 1-p_neu) 或 (1-p_neu, p_neu, 0)，p_neu ≥ 0.5
本文件按此规则实现（单元测试 code/tests/test_neutral.py 会用公开预测文件核对这些性质）。

设 p = P(positive)，h = margin/2：
  p < 0.5-h         -> 负，概率 (1-p, 0, p)
  p > 0.5+h         -> 正，概率 (1-p, 0, p)
  否则（中性带内）  -> 中，p_neu = 1 - |p-0.5|/margin，剩余概率给 p 所在的那一侧
margin=0 表示不设中性带（p ≥ 0.5 为正，否则为负），用作消融。
"""
import numpy as np

PAPER_MARGIN = 1.0 / 3.0
NEG, NEU, POS = 0, 1, 2


def binary_to_three_way(p_pos, margin=PAPER_MARGIN):
    """p_pos: 形如 [N] 的 P(positive)。返回 (preds[N] ∈ {0,1,2}, probs[N,3] 依次为 负/中/正)。"""
    p = np.asarray(p_pos, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError("p_pos 必须是一维数组")
    if not (0.0 <= margin < 1.0):
        raise ValueError("margin 应在 [0, 1) 内")
    probs = np.zeros((p.shape[0], 3), dtype=np.float64)
    preds = np.zeros(p.shape[0], dtype=np.int64)
    if margin == 0.0:
        pos = p >= 0.5
        neg = ~pos
        neu = np.zeros_like(pos)
    else:
        h = margin / 2.0
        neg = p < 0.5 - h
        pos = p > 0.5 + h
        neu = ~(neg | pos)
    for m in (neg, pos):
        probs[m, NEG] = 1.0 - p[m]
        probs[m, POS] = p[m]
    preds[neg] = NEG
    preds[pos] = POS
    if neu.any():
        p_neu = 1.0 - np.abs(p[neu] - 0.5) / margin
        probs[neu, NEU] = p_neu
        side_pos = p[neu] >= 0.5
        rest = 1.0 - p_neu
        probs[np.where(neu)[0][side_pos], POS] = rest[side_pos]
        probs[np.where(neu)[0][~side_pos], NEG] = rest[~side_pos]
        preds[neu] = NEU
    return preds, probs


def format_pred_lines(preds, probs):
    """checklist 的 pred_and_softmax 格式：'pred p_neg p_neu p_pos'。
    概率按 float 完整精度写出（官方文件是 6 位小数；这里不做舍入，避免改变 INV/DIR 的阈值判断）。"""
    return ["%d %r %r %r" % (int(c), float(a), float(b), float(d)) for c, (a, b, d) in zip(preds, probs)]
