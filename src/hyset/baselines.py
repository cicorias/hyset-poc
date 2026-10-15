"""Independent-scoring baselines (the "semantic matching" paradigm of Sec. 2.3)."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .data import ToolLibrary

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(_tokens(d)) for d in docs]
        self.lens = np.array([sum(d.values()) for d in self.docs], dtype=np.float32)
        self.avg = float(self.lens.mean()) if len(self.lens) else 0.0
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.lens / max(self.avg, 1e-9))
        for t in set(_tokens(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            tf = np.array([d.get(t, 0) for d in self.docs], dtype=np.float32)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out


def bm25_rankings(library: ToolLibrary, queries: list[str], k: int) -> list[list[int]]:
    index = BM25(library.texts())
    return [np.argsort(-index.scores(q), kind="stable")[:k].tolist() for q in queries]


def dense_rankings(tool_emb: np.ndarray, query_emb: np.ndarray, k: int) -> list[list[int]]:
    """Frozen bi-encoder: cosine similarity between query and tool-description embeddings."""
    sims = query_emb @ tool_emb.T
    k = min(k, sims.shape[1])
    top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(sims, top, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(top, order, axis=1).tolist()
