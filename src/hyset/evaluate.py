"""Offline evaluation of a trained model and the independent-scoring baselines."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from torch import Tensor

from .inference import InferenceConfig, SetPrediction, SetRetriever
from .metrics import rank_metrics, set_metrics
from .model import HySet


def evaluate_model(
    model: HySet,
    infer_cfg: InferenceConfig,
    r: Tensor,
    gts: Sequence[Sequence[int]],
    ks: Sequence[int] = (3, 5),
) -> tuple[dict[str, dict[str, float]], list[SetPrediction]]:
    """Rows:
    - ``hyset``: greedy set-aware ranking for rank metrics + argmax set Ê for set metrics
    - ``hyset-stage1``: the same trained parameters ranking tools independently (Eq. 12)
    """
    retriever = SetRetriever(model, infer_cfg)
    preds = retriever.predict(r, rank_k=max(ks))
    rows = {
        "hyset": {
            **rank_metrics([p.ranking for p in preds], gts, ks),
            **set_metrics([p.tool_set for p in preds], gts),
        },
        "hyset-stage1": rank_metrics(retriever.stage1_rank(r, max(ks)), gts, ks),
    }
    return rows, preds


def topk_set_metrics(
    rankings: Sequence[Sequence[int]], gts: Sequence[Sequence[int]], k: int
) -> dict[str, float]:
    """Set metrics for a fixed-budget retriever that hands its top-k to the agent."""
    return set_metrics([r[:k] for r in rankings], gts)


def subset_rows(
    rankings_by_row: dict[str, list[list[int]]],
    sets_by_row: dict[str, list[list[int]]],
    gts: Sequence[Sequence[int]],
    mask: np.ndarray,
    ks: Sequence[int] = (3, 5),
) -> dict[str, dict[str, float]]:
    """Recompute rows on a boolean subset of queries (e.g. unseen tool sets)."""
    idx = np.flatnonzero(mask)
    sub_gts = [gts[i] for i in idx]
    out = {}
    for name, rankings in rankings_by_row.items():
        row = rank_metrics([rankings[i] for i in idx], sub_gts, ks)
        if name in sets_by_row:
            row |= set_metrics([sets_by_row[name][i] for i in idx], sub_gts)
        out[name] = row
    return out
