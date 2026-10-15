"""Frozen text encoders r(.) for queries and tool descriptions.

Specs accepted by :func:`make_encoder`:

- ``"st:<hf-model-id>"`` or a bare HF id: a sentence-transformers model (default
  ``st:BAAI/bge-small-en-v1.5``), run on CPU.
- ``"hashing"`` or ``"hashing:<dim>"``: deterministic signed feature hashing of word
  unigrams + bigrams. No download; low quality. Used by tests and quick demos.

Every encoder returns L2-normalized float32 arrays. :class:`CachedEncoder` memoizes
results on disk keyed by the encoder spec and the exact input texts.
"""

from __future__ import annotations

import hashlib
import re
import zlib
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

Kind = Literal["query", "tool"]

DEFAULT_ENCODER = "st:BAAI/bge-small-en-v1.5"

# bge v1.5 retrieval convention: instruct queries, leave passages bare.
_QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


class Encoder(Protocol):
    spec: str
    dim: int

    def encode(self, texts: list[str], kind: Kind) -> np.ndarray: ...


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


class HashingEncoder:
    _token = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.spec = f"hashing:{dim}"

    def _features(self, text: str) -> list[str]:
        words = self._token.findall(text.lower())
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]

    def encode(self, texts: list[str], kind: Kind) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for feat in self._features(text):
                h = zlib.crc32(feat.encode())
                out[i, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        return _l2norm(np.sign(out) * np.log1p(np.abs(out)))


class SentenceTransformerEncoder:
    def __init__(self, model_id: str, device: str = "cpu", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.spec = f"st:{model_id}"
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_id, device=device)
        get_dim = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        self.dim = int(get_dim())
        self.query_prefix = _QUERY_PREFIXES.get(model_id, "")

    def encode(self, texts: list[str], kind: Kind) -> np.ndarray:
        if kind == "query" and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        emb = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 2000,
        )
        return np.asarray(emb, dtype=np.float32)


class CachedEncoder:
    """Disk memoization wrapper; safe to share a cache dir across runs."""

    def __init__(self, inner: Encoder, cache_dir: Path | None):
        self.inner = inner
        self.spec = inner.spec
        self.dim = inner.dim
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def encode(self, texts: list[str], kind: Kind) -> np.ndarray:
        if self.cache_dir is None or len(texts) < 32:
            return self.inner.encode(texts, kind)
        h = hashlib.sha1(f"{self.spec}\x00{kind}\x00".encode())
        for t in texts:
            h.update(t.encode())
            h.update(b"\x00")
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.spec)
        path = self.cache_dir / slug / f"{kind}-{h.hexdigest()[:16]}.npy"
        if path.exists():
            return np.load(path)
        emb = self.inner.encode(texts, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, emb)
        return emb


def make_encoder(spec: str = DEFAULT_ENCODER, cache_dir: Path | None = None) -> Encoder:
    if spec.startswith("hashing"):
        _, _, dim = spec.partition(":")
        inner: Encoder = HashingEncoder(int(dim) if dim else 512)
    else:
        model_id = spec.removeprefix("st:")
        inner = SentenceTransformerEncoder(model_id)
    return CachedEncoder(inner, cache_dir) if cache_dir else inner
