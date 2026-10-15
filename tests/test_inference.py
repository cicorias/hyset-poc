import itertools

import torch

from hyset.inference import InferenceConfig, SetRetriever, n_candidates
from hyset.model import HySet, ModelConfig


def make(V=30, d=12, M=4, seed=0):
    torch.manual_seed(seed)
    m = HySet(V, d, ModelConfig(max_card=M, align_scale=3.0), tool_init=torch.randn(V, d))
    with torch.no_grad():
        m.A.normal_(0, 0.4)
    return m


def brute_force_best(m, r, shortlist, M):
    cands = [c for k in range(1, M + 1) for c in itertools.combinations(shortlist, k)]
    sets = torch.full((1, len(cands), M), -1, dtype=torch.long)
    for j, c in enumerate(cands):
        sets[0, j, : len(c)] = torch.tensor(c)
    f = m.score_sets(r, sets)[0].detach()
    return set(cands[int(f.argmax())]), float(f.max())


def test_paper_candidate_count():
    assert n_candidates(20, 5) == 21_699  # K_pool=20, M=5 as reported in the paper


def test_exhaustive_matches_brute_force():
    m = make()
    ret = SetRetriever(m, InferenceConfig(k1=5, k_pool=8))
    r = torch.randn(4, 12)
    S = ret.shortlist(r)
    best, score = ret.best_sets(r, S)
    for b in range(4):
        want, want_score = brute_force_best(m, r[b : b + 1], S[b].tolist(), 4)
        assert {t for t in best[b].tolist() if t >= 0} == want
        assert abs(float(score[b]) - want_score) < 1e-4


def test_top_sets_sorted_and_consistent_with_best():
    m = make()
    ret = SetRetriever(m, InferenceConfig(k1=5, k_pool=8))
    r = torch.randn(2, 12)
    S = ret.shortlist(r)
    best, _ = ret.best_sets(r, S)
    tops = ret.top_sets(r, S, 10)
    for b in range(2):
        assert set(tops[b][0]) == {t for t in best[b].tolist() if t >= 0}
        assert len(set(tops[b])) == len(tops[b]) == 10


def test_greedy_rank_matches_marginal_gain():
    m = make()
    ret = SetRetriever(m, InferenceConfig(k1=5, k_pool=8))
    r = torch.randn(1, 12)
    S = ret.shortlist(r)
    rank = ret.greedy_rank(r, S, 6)[0].tolist()
    shortlist = S[0].tolist()
    chosen = []
    for _ in range(4):  # up to M; beyond that the tail is filled by stage-1 score
        cands = [t for t in shortlist if t not in chosen]
        sets = torch.full((1, len(cands), 4), -1, dtype=torch.long)
        for j, t in enumerate(cands):
            row = [*chosen, t]
            sets[0, j, : len(row)] = torch.tensor(row)
        chosen.append(cands[int(m.score_sets(r, sets)[0].argmax())])
    assert rank[:4] == chosen
    assert len(rank) == 6 and len(set(rank)) == 6


def test_expansion_adds_complementary_tool():
    """A tool irrelevant on its own but strongly complementary to the top-K1 enters S."""
    V, d = 10, 4
    Z = torch.eye(V, d) + 0.0  # tools 0..3 on axes; others zero -> fix below
    Z = torch.randn(V, d)
    Z[0] = torch.tensor([1.0, 0, 0, 0])
    Z[9] = torch.tensor([0, 0, 0, 1.0])  # orthogonal to the query
    m = HySet(V, d, ModelConfig(max_card=2, align_scale=1.0), tool_init=Z)
    with torch.no_grad():
        A = torch.zeros(d, d)
        A[0, 3] = A[3, 0] = 50.0  # tool 9 pairs strongly with tool 0
        m.A[0] = A
    r = torch.tensor([[1.0, 0, 0, 0]])
    base = SetRetriever(m, InferenceConfig(k1=2, k_pool=3, expand=False)).shortlist(r)[0]
    expanded = SetRetriever(m, InferenceConfig(k1=2, k_pool=3, expand=True)).shortlist(r)[0]
    assert 9 not in base.tolist()
    assert 9 in expanded.tolist()


def test_predict_returns_valid_sets():
    m = make()
    preds = SetRetriever(m, InferenceConfig(k1=5, k_pool=8, rank_k=5)).predict(torch.randn(3, 12))
    for p in preds:
        assert 1 <= len(p.tool_set) <= 4
        assert set(p.tool_set) <= set(p.shortlist)
        assert len(p.ranking) == 5
