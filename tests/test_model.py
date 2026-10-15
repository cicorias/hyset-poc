import itertools

import pytest
import torch

from hyset.model import HySet, ModelConfig


def make(interaction="cardinality", V=12, d=8, M=4, seed=0):
    torch.manual_seed(seed)
    m = HySet(V, d, ModelConfig(max_card=M, interaction=interaction, align_scale=2.0),
              tool_init=torch.randn(V, d))  # fmt: skip
    with torch.no_grad():
        m.A.normal_(0, 0.5)
    return m


def sets_tensor(sets, width):
    t = torch.full((1, len(sets), width), -1, dtype=torch.long)
    for j, s in enumerate(sets):
        t[0, j, : len(s)] = torch.tensor(s)
    return t


def test_singleton_score_is_tool_score():
    m = make()
    r = torch.randn(1, 8)
    f = m.score_sets(r, sets_tensor([(t,) for t in range(12)], 4))[0]
    assert torch.allclose(f, m.tool_scores(r)[0], atol=1e-5)


def test_hand_computed_score():
    m = make()
    r = torch.randn(1, 8)
    s = (1, 4, 7)
    Z, mats = m.Z, m.interaction_matrices()
    ell = torch.stack([r[0] @ m.P @ Z[t] for t in s])
    align = (torch.softmax(ell, 0) * ell).sum()
    pairs = sum(Z[a] @ mats[3] @ Z[b] for a, b in itertools.combinations(s, 2))
    f = m.score_sets(r, sets_tensor([s], 4))[0, 0]
    assert torch.allclose(f, align + pairs, atol=1e-5)


def test_permutation_and_padding_invariance():
    m = make()
    r = torch.randn(1, 8)
    sets = [(2, 5, 9), (9, 2, 5), (5, 9, 2)]
    f_narrow = m.score_sets(r, sets_tensor(sets, 3))[0]
    f_wide = m.score_sets(r, sets_tensor(sets, 4))[0]
    assert torch.allclose(f_narrow, f_narrow[0].expand(3), atol=1e-5)
    assert torch.allclose(f_narrow, f_wide, atol=1e-5)


def test_interaction_matrices_symmetric_and_unit_norm_Z():
    m = make()
    mats = m.interaction_matrices()
    assert torch.allclose(mats, mats.mT)
    assert torch.allclose(mats[:2], torch.zeros_like(mats[:2]))
    assert torch.allclose(m.Z.norm(dim=-1), torch.ones(12), atol=1e-5)


def test_cardinality_changes_pair_contribution():
    m = make()
    _, f_set_3, _ = m.score_sets(torch.zeros(1, 8), sets_tensor([(0, 1, 2)], 4), True)
    with torch.no_grad():
        m.A[1] = m.A[0]  # make M_3 == M_2
    _, f_set_3b, _ = m.score_sets(torch.zeros(1, 8), sets_tensor([(0, 1, 2)], 4), True)
    assert not torch.allclose(f_set_3, f_set_3b)


def _mobius(f, E):
    return sum((-1) ** (len(E) - k) * f(T) for k in range(len(E) + 1)
               for T in itertools.combinations(E, k))  # fmt: skip


@pytest.mark.parametrize("interaction,expect_zero", [("shared", True), ("cardinality", False)])
def test_theorem1_higher_order_mobius(interaction, expect_zero):
    """F_set is pairwise-decomposable (no order>=3 Möbius terms) iff all M_m agree."""
    m = make(interaction)

    def f_set(T):
        if len(T) < 2:
            return 0.0
        _, fs, _ = m.score_sets(torch.zeros(1, 8), sets_tensor([T], 4), True)
        return float(fs[0, 0].detach())

    coeffs = [_mobius(f_set, E) for E in itertools.combinations(range(6), 3)]
    if expect_zero:
        assert max(abs(c) for c in coeffs) < 1e-4
    else:
        assert max(abs(c) for c in coeffs) > 1e-3


def test_none_interaction_has_no_set_term():
    m = make("none")
    _, f_set, _ = m.score_sets(torch.randn(1, 8), sets_tensor([(0, 1, 2, 3)], 4), True)
    assert float(f_set.abs().sum()) == 0.0
    assert float(m.regularizer()) == 0.0


def test_gradients_reach_all_parameters():
    m = make()
    f = m.score_sets(torch.randn(2, 8), torch.tensor([[[0, 1, 2, -1], [3, -1, -1, -1]]] * 2))
    (f.sum() + m.regularizer()).backward()
    for name in ("z_raw", "P", "A"):
        g = getattr(m, name).grad
        assert g is not None and g.abs().sum() > 0, name


def test_rejects_oversized_sets():
    m = make(M=3)
    with pytest.raises(ValueError):
        m.score_sets(torch.randn(1, 8), sets_tensor([(0, 1, 2, 3)], 4))
