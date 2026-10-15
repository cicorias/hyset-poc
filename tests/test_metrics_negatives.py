import math

import torch

from hyset.metrics import comp_at_k, ndcg_at_k, rank_metrics, recall_at_k, set_metrics
from hyset.negatives import NegativeMix, NegativeSampler, pools_to_tensor


def test_rank_metrics():
    ranking, gt = [3, 1, 7, 2, 9], {1, 2, 5}
    assert recall_at_k(ranking, gt, 3) == 1 / 3
    assert recall_at_k(ranking, gt, 5) == 2 / 3
    assert comp_at_k(ranking, gt, 5) == 0.0
    assert comp_at_k([1, 2, 5, 0], gt, 3) == 1.0
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert abs(ndcg_at_k(ranking, gt, 5) - dcg / idcg) < 1e-9
    agg = rank_metrics([ranking, [1, 2, 5]], [gt, gt], ks=(3,))
    assert abs(agg["comp@3"] - 0.5) < 1e-9


def test_set_metrics():
    m = set_metrics([[1, 2], [1, 2, 3, 4]], [[1, 2], [1, 2]])
    assert m["set_exact"] == 0.5 and m["set_cover"] == 1.0 and m["set_size"] == 3.0
    assert abs(m["set_precision"] - 0.75) < 1e-9


def test_paper_quotas_for_63_negatives():
    s = NegativeSampler(100, 64, 5, NegativeMix.paper())
    assert (s.n_in_batch, s.n_hard) == (19, 13)  # size-matched fills the other 31


def test_pool_distinct_gt_first():
    s = NegativeSampler(50, 32, 5, NegativeMix(model=0.1), seed=3)
    s.refresh_neighbors(torch.nn.functional.normalize(torch.randn(50, 8), dim=-1))
    gt = (4, 9, 11)
    mined = [(4, 9), (4, 9, 11), (1, 4, 9, 11)]
    pool = s.pool(gt, [(1, 2), (4, 9, 11), (7,)], mined=mined)
    assert pool[0] == gt
    assert len(pool) == 32 and len(set(pool)) == 32
    assert (4, 9) in pool and (1, 4, 9, 11) in pool  # mined negatives used
    assert all(1 <= len(p) <= 5 and list(p) == sorted(p) for p in pool)


def test_tiny_library_still_fills_pool():
    s = NegativeSampler(6, 20, 3, seed=0)
    pool = s.pool((0, 1), [])
    assert len(pool) == 20 and len(set(pool)) == 20


def test_pools_to_tensor_pads():
    sets, valid = pools_to_tensor([[(1, 2), (3,)], [(4,)]], 3)
    assert sets.shape == (2, 2, 3)
    assert valid.tolist() == [[True, True], [True, False]]
    assert sets[0, 0].tolist() == [1, 2, -1]
