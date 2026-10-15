import numpy as np
import pytest

from hyset.encoders import CachedEncoder, HashingEncoder, make_encoder


def test_hashing_encoder_normalized_and_deterministic():
    enc = HashingEncoder(128)
    a = enc.encode(["find flights to Tokyo", "book a hotel"], "query")
    b = enc.encode(["find flights to Tokyo", "book a hotel"], "query")
    assert a.shape == (2, 128) and a.dtype == np.float32
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(a, b)
    assert a[0] @ enc.encode(["flights to Tokyo please"], "query")[0] > a[0] @ a[1]


def test_cached_encoder_roundtrip(tmp_path):
    texts = [f"tool number {i}" for i in range(40)]
    enc = CachedEncoder(HashingEncoder(64), tmp_path)
    first = enc.encode(texts, "tool")
    assert len(list(tmp_path.rglob("*.npy"))) == 1
    assert np.array_equal(first, enc.encode(texts, "tool"))
    assert make_encoder("hashing:32").dim == 32


@pytest.mark.slow
def test_sentence_transformer_encoder():
    enc = make_encoder("st:BAAI/bge-small-en-v1.5")
    q = enc.encode(["convert dollars to yen"], "query")
    t = enc.encode(["Convert amounts between currencies", "Weather forecasts"], "tool")
    assert enc.dim == 384 and q.shape == (1, 384)
    assert np.allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-4)
    assert q[0] @ t[0] > q[0] @ t[1]
