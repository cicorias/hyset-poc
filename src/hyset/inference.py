"""Two-stage set-level inference (paper Sec. 2.4, "Set-Level Inference").

Stage 1 (Eq. 12-13): score every tool alone, keep the top K1 as S0, then add the
K_pool - K1 tools with the highest learned complementarity to S0,
g(t_j) = max_{t_i in S0} z_j^T M_M z_i.

Stage 2 (Eq. 14): enumerate every subset of the shortlist S with 1..M tools and return
the argmax of F. With K_pool=20, M=5 that is 21,699 candidates per query; all of them
are scored at once from the per-cardinality Gram matrices Z_S M_m Z_S^T.

A second output, the greedy marginal-gain ranking A_k = A_{k-1} ∪ {argmax_t F(A_{k-1} ∪ {t})},
feeds the rank-based metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import torch
from torch import Tensor

from .model import HySet


@dataclass
class InferenceConfig:
    k1: int = 15
    k_pool: int = 20
    expand: bool = True  # complementarity expansion of the shortlist (Eq. 13)
    rank_k: int = 5  # length of the greedy ranking
    batch_size: int = 64


@lru_cache(maxsize=8)
def enumerate_subsets(k_pool: int, max_card: int) -> list[tuple[int, Tensor, Tensor, Tensor]]:
    """Per cardinality m: (m, combos [N_m, m], pair_a [P], pair_b [P]) over range(k_pool)."""
    groups = []
    for m in range(1, min(max_card, k_pool) + 1):
        combos = torch.tensor(list(combinations(range(k_pool), m)), dtype=torch.long)
        pairs = list(combinations(range(m), 2))
        pa = torch.tensor([a for a, _ in pairs], dtype=torch.long)
        pb = torch.tensor([b for _, b in pairs], dtype=torch.long)
        groups.append((m, combos, pa, pb))
    return groups


def n_candidates(k_pool: int, max_card: int) -> int:
    return sum(len(c) for _, c, _, _ in enumerate_subsets(k_pool, max_card))


@dataclass
class SetPrediction:
    tool_set: list[int]  # Ê(x): the argmax hyperedge, passed to the agent
    set_score: float
    ranking: list[int]  # greedy marginal-gain ranking, for rank metrics
    shortlist: list[int]


class SetRetriever:
    def __init__(self, model: HySet, cfg: InferenceConfig | None = None):
        self.model = model
        self.cfg = cfg or InferenceConfig()
        if self.cfg.k_pool < model.cfg.max_card:
            raise ValueError("k_pool must be >= max_card")

    # ----------------------------------------------------------------- stage 1

    @torch.no_grad()
    def shortlist(self, r: Tensor) -> Tensor:
        model, cfg = self.model, self.cfg
        logits = model.tool_scores(r)  # [B, V]
        kp = min(cfg.k_pool, model.n_tools)
        k1 = min(cfg.k1, kp)
        if not cfg.expand or not model.has_interaction or k1 == kp:
            return logits.topk(kp, dim=-1).indices
        s0 = logits.topk(k1, dim=-1).indices  # [B, K1]
        Z = model.Z
        zm = Z @ model.interaction_matrices()[model.cfg.max_card]  # [V, d]
        g = torch.einsum("vd,bkd->bvk", zm, Z[s0]).amax(-1)  # [B, V]
        g.scatter_(1, s0, float("-inf"))
        s1 = g.topk(kp - k1, dim=-1).indices
        return torch.cat([s0, s1], dim=1)

    # ----------------------------------------------------------------- stage 2

    def _shortlist_terms(self, r: Tensor, S: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        model = self.model
        zS = model.Z[S]  # [B, Kp, d]
        ell = torch.einsum("bkd,bd->bk", zS, model.query_proj(r))  # [B, Kp]
        mats = model.interaction_matrices()
        # Gram matrices G_m = Z_S M_m Z_S^T for every cardinality: [M+1, B, Kp, Kp]
        gram = torch.einsum("bid,mde,bje->mbij", zS, mats, zS)
        return zS, ell, gram

    @torch.no_grad()
    def best_sets(self, r: Tensor, S: Tensor) -> tuple[Tensor, Tensor]:
        """Exhaustive argmax of F over all 1..M subsets of each shortlist.

        Returns (best [B, M] global tool indices padded with -1, best_score [B])."""
        model = self.model
        _, ell, gram = self._shortlist_terms(r, S)
        B, kp = ell.shape
        M = model.cfg.max_card
        best_score = torch.full((B,), float("-inf"))
        best_local = torch.full((B, M), -1, dtype=torch.long)
        for m, combos, pa, pb in enumerate_subsets(kp, M):
            e = ell[:, combos]  # [B, N, m]
            f = (torch.softmax(e, dim=-1) * e).sum(-1)
            if m >= 2 and model.has_interaction:
                f = f + gram[m][:, combos[:, pa], combos[:, pb]].sum(-1)
            score, j = f.max(dim=-1)
            better = score > best_score
            best_score = torch.where(better, score, best_score)
            chosen = torch.full((B, M), -1, dtype=torch.long)
            chosen[:, :m] = combos[j]
            best_local = torch.where(better[:, None], chosen, best_local)
        best = torch.where(best_local >= 0, S.gather(1, best_local.clamp(min=0)), best_local)
        return best, best_score

    @torch.no_grad()
    def top_sets(self, r: Tensor, S: Tensor, n: int) -> list[list[tuple[int, ...]]]:
        """The n highest-scoring hyperedges per query over all 1..M subsets of S (global
        tool ids, best first). Used to mine model-based negatives during training."""
        model = self.model
        _, ell, gram = self._shortlist_terms(r, S)
        B, kp = ell.shape
        scores, sets = [], []
        for m, combos, pa, pb in enumerate_subsets(kp, model.cfg.max_card):
            e = ell[:, combos]
            f = (torch.softmax(e, dim=-1) * e).sum(-1)
            if m >= 2 and model.has_interaction:
                f = f + gram[m][:, combos[:, pa], combos[:, pb]].sum(-1)
            s, j = f.topk(min(n, f.shape[1]), dim=-1)
            padded = torch.full((B, s.shape[1], model.cfg.max_card), -1, dtype=torch.long)
            padded[..., :m] = combos[j]
            scores.append(s)
            sets.append(padded)
        scores_t, sets_t = torch.cat(scores, 1), torch.cat(sets, 1)
        order = scores_t.topk(min(n, scores_t.shape[1]), dim=-1).indices
        out = []
        for b in range(B):
            rows = []
            for local in sets_t[b, order[b]].tolist():
                rows.append(tuple(sorted(S[b, i].item() for i in local if i >= 0)))
            out.append(rows)
        return out

    @torch.no_grad()
    def greedy_rank(self, r: Tensor, S: Tensor, k: int) -> Tensor:
        """Greedy marginal-gain ordering of the shortlist under F: [B, min(k, Kp)].

        Past M tools F is undefined, so the tail is filled by stage-1 score."""
        model = self.model
        _, ell, gram = self._shortlist_terms(r, S)
        B, kp = ell.shape
        k = min(k, kp)
        M = model.cfg.max_card
        w = torch.exp(ell - ell.amax(-1, keepdim=True))  # stable softmax weights
        selected = torch.zeros(B, kp, dtype=torch.bool)
        num = torch.zeros(B)  # sum_{a in A} w_a * l_a
        den = torch.zeros(B)  # sum_{a in A} w_a
        order = []
        for step in range(k):
            m = step + 1
            if m > M:
                rest = ell.masked_fill(selected, float("-inf")).topk(k - step, dim=-1).indices
                order.extend(rest.unbind(-1))
                break
            f = (num[:, None] + w * ell) / (den[:, None] + w)
            if m >= 2 and model.has_interaction:
                G = gram[m]
                s = selected.float()
                within = 0.5 * (
                    torch.einsum("bi,bij,bj->b", s, G, s) - (G.diagonal(dim1=1, dim2=2) * s).sum(-1)
                )
                cross = torch.einsum("bij,bj->bi", G, s)
                f = f + within[:, None] + cross
            pick = f.masked_fill(selected, float("-inf")).argmax(-1)
            rows = torch.arange(B)
            selected[rows, pick] = True
            num += w[rows, pick] * ell[rows, pick]
            den += w[rows, pick]
            order.append(pick)
        local = torch.stack(order, dim=1)
        return S.gather(1, local)

    # ----------------------------------------------------------------- API

    @torch.no_grad()
    def predict(self, r: Tensor, rank_k: int | None = None) -> list[SetPrediction]:
        rank_k = rank_k or self.cfg.rank_k
        out: list[SetPrediction] = []
        for start in range(0, r.shape[0], self.cfg.batch_size):
            rb = r[start : start + self.cfg.batch_size]
            S = self.shortlist(rb)
            best, score = self.best_sets(rb, S)
            ranking = self.greedy_rank(rb, S, rank_k)
            for i in range(rb.shape[0]):
                out.append(
                    SetPrediction(
                        tool_set=[t for t in best[i].tolist() if t >= 0],
                        set_score=float(score[i]),
                        ranking=ranking[i].tolist(),
                        shortlist=S[i].tolist(),
                    )
                )
        return out

    @torch.no_grad()
    def stage1_rank(self, r: Tensor, k: int) -> list[list[int]]:
        """Independent per-tool ranking from Eq. 12 alone (no set scoring)."""
        return self.model.tool_scores(r).topk(min(k, self.model.n_tools), dim=-1).indices.tolist()
