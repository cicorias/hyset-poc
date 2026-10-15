"""Retrieval metrics.

Rank metrics take a ranked list of tool indices and the ground-truth set E*:

- Recall@K: |top-K ∩ E*| / |E*|
- NDCG@K:   binary-relevance NDCG, ideal DCG uses min(|E*|, K) hits
- COMP@K:   1 if E* ⊆ top-K else 0  (completeness, Qu et al. 2024)

Set metrics score the variable-size predicted set Ê that is handed to the agent.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence


def recall_at_k(ranking: Sequence[int], gt: Collection[int], k: int) -> float:
    return len(set(ranking[:k]) & set(gt)) / len(gt)


def ndcg_at_k(ranking: Sequence[int], gt: Collection[int], k: int) -> float:
    gt = set(gt)
    dcg = sum(1.0 / math.log2(i + 2) for i, t in enumerate(ranking[:k]) if t in gt)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt), k)))
    return dcg / idcg


def comp_at_k(ranking: Sequence[int], gt: Collection[int], k: int) -> float:
    return float(set(gt) <= set(ranking[:k]))


def rank_metrics(
    rankings: Sequence[Sequence[int]],
    gts: Sequence[Collection[int]],
    ks: Sequence[int] = (3, 5),
) -> dict[str, float]:
    n = len(rankings)
    out: dict[str, float] = {}
    pairs = list(zip(rankings, gts, strict=True))
    for k in ks:
        out[f"recall@{k}"] = sum(recall_at_k(r, g, k) for r, g in pairs) / n
        out[f"ndcg@{k}"] = sum(ndcg_at_k(r, g, k) for r, g in pairs) / n
        out[f"comp@{k}"] = sum(comp_at_k(r, g, k) for r, g in pairs) / n
    return out


def set_metrics(
    preds: Sequence[Collection[int]], gts: Sequence[Collection[int]]
) -> dict[str, float]:
    n = len(preds)
    exact = cover = prec = rec = f1 = size = 0.0
    for p, g in zip(preds, gts, strict=True):
        p, g = set(p), set(g)
        hit = len(p & g)
        pr = hit / len(p) if p else 0.0
        rc = hit / len(g)
        exact += float(p == g)
        cover += float(g <= p)
        prec += pr
        rec += rc
        f1 += 2 * pr * rc / (pr + rc) if hit else 0.0
        size += len(p)
    return {
        "set_exact": exact / n,
        "set_cover": cover / n,
        "set_precision": prec / n,
        "set_recall": rec / n,
        "set_f1": f1 / n,
        "set_size": size / n,
    }
