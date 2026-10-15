"""The HySet scoring family F_theta(x, E) = F_set(E) + F_align(x, E).

Paper references (arXiv:2607.25718, Sec. 2.4):

- Eq. (3)  F_set(E)    = sum_{a<b} z_a^T M_m z_b, with m = |E| and M_m symmetric.
- Eq. (4-6) F_align(x,E) = sum_k alpha_k * l(x, t_k), with l(x, t) = r(x)^T P z_t and
  alpha = softmax over the tools of E (cross-attention pooling, query = r(x)).
- Eq. (11) unit-norm tool embeddings ||z_j|| = 1.
- Eq. (12) singleton score F(x, {t}) = r(x)^T P z_t (F_set is empty for one tool).

Trainable parameters theta = (Z, {M_m}_{m=2..M}, P). The query encoder r(.) is frozen and
lives outside this module; callers pass precomputed query embeddings ``r``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

Interaction = Literal["cardinality", "shared", "identity", "none"]


@dataclass
class ModelConfig:
    max_card: int = 5  # M: largest admissible tool-set size
    d_z: int | None = None  # tool-embedding width; None -> encoder width d_r
    # "cardinality": one learned M_m per set size (the paper's model)
    # "shared":      one learned M for all sizes        (ablation "Shared M")
    # "identity":    M_m = I                            (ablation "M_m = I")
    # "none":        F_set = 0                          (ablation "w/o F_set")
    interaction: Interaction = "cardinality"
    # P is initialized to align_scale * (projection), so l(x, t) starts as a scaled
    # cosine similarity between the frozen query and tool embeddings. It also sets the
    # sharpness of the in-set attention softmax; see docs/DESIGN.md section 4.
    align_scale: float = 5.0
    seed: int = 0


class HySet(nn.Module):
    def __init__(
        self,
        n_tools: int,
        d_r: int,
        cfg: ModelConfig,
        tool_init: np.ndarray | Tensor | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.n_tools = n_tools
        self.d_r = d_r
        self.d_z = d_z = cfg.d_z or d_r
        gen = torch.Generator().manual_seed(cfg.seed)

        if d_z == d_r:
            proj = torch.eye(d_r)
        else:  # random orthonormal map between the two spaces
            q, _ = torch.linalg.qr(torch.randn(max(d_r, d_z), min(d_r, d_z), generator=gen))
            proj = q if d_r >= d_z else q.T

        if tool_init is None:
            z0 = torch.randn(n_tools, d_z, generator=gen)
        else:
            z0 = torch.as_tensor(np.asarray(tool_init), dtype=torch.float32) @ proj
        if z0.shape != (n_tools, d_z):
            raise ValueError(f"tool_init has shape {tuple(z0.shape)}, want ({n_tools}, {d_r})")

        self.z_raw = nn.Parameter(F.normalize(z0, dim=-1))
        self.P = nn.Parameter(cfg.align_scale * proj)  # [d_r, d_z]
        n_mats = {"cardinality": cfg.max_card - 1, "shared": 1}.get(cfg.interaction, 0)
        # Unconstrained A_m; the symmetric M_m = (A_m + A_m^T) / 2. Zero init means the
        # model starts as a pure alignment scorer and learns interactions from data.
        self.A = nn.Parameter(torch.zeros(n_mats, d_z, d_z))

    # ----------------------------------------------------------------- parameters

    @property
    def Z(self) -> Tensor:
        """Unit-norm tool embeddings (Eq. 11), [V, d_z]."""
        return F.normalize(self.z_raw, dim=-1)

    @property
    def has_interaction(self) -> bool:
        return self.cfg.interaction != "none"

    def interaction_matrices(self) -> Tensor:
        """Stack of M_m indexed by cardinality m: [M+1, d_z, d_z]; rows 0 and 1 are zero."""
        M, d = self.cfg.max_card, self.d_z
        lead = self.A.new_zeros(2, d, d)
        kind = self.cfg.interaction
        if kind == "cardinality":
            mats = (self.A + self.A.mT) / 2
        elif kind == "shared":
            mats = ((self.A[0] + self.A[0].T) / 2).expand(M - 1, d, d)
        elif kind == "identity":
            mats = torch.eye(d, device=self.A.device).expand(M - 1, d, d)
        else:
            mats = self.A.new_zeros(M - 1, d, d)
        return torch.cat([lead, mats], dim=0)

    def regularizer(self) -> Tensor:
        """sum_m ||M_m||_F^2 over the learned interaction matrices (Eq. 10)."""
        if self.A.numel() == 0:
            return self.A.new_zeros(())
        return (((self.A + self.A.mT) / 2) ** 2).sum()

    # ----------------------------------------------------------------- scoring

    def query_proj(self, r: Tensor) -> Tensor:
        """r(x)^T P, the query mapped into tool space: [B, d_z]."""
        return r @ self.P

    def tool_scores(self, r: Tensor) -> Tensor:
        """Per-tool scores l(x, t) = r^T P z_t for every tool (Eq. 12): [B, V]."""
        return self.query_proj(r) @ self.Z.T

    def score_sets(self, r: Tensor, sets: Tensor, return_parts: bool = False):
        """Score candidate hyperedges.

        Args:
            r: frozen query embeddings, [B, d_r].
            sets: tool indices, [B, K, W] padded with -1 (W <= max_card). Every row
                must contain at least one tool.
        Returns:
            F [B, K], or (F, F_set, F_align) when ``return_parts``.
        """
        mask = sets >= 0
        card = mask.sum(-1)
        if int(card.max()) > self.cfg.max_card:
            raise ValueError("candidate set larger than max_card")
        z = self.Z[sets.clamp(min=0)]  # [B, K, W, d]

        # F_align: attention-pooled per-tool matching scores (Eq. 5-6).
        ell = torch.einsum("bkwd,bd->bkw", z, self.query_proj(r))
        alpha = torch.softmax(ell.masked_fill(~mask, float("-inf")), dim=-1)
        f_align = (alpha.nan_to_num(0.0) * ell.masked_fill(~mask, 0.0)).sum(-1)

        f_set = self._f_set(z, mask, card)
        f = f_set + f_align
        return (f, f_set, f_align) if return_parts else f

    def _f_set(self, z: Tensor, mask: Tensor, card: Tensor) -> Tensor:
        if not self.has_interaction:
            return z.new_zeros(card.shape)
        mats = self.interaction_matrices()
        zm = torch.zeros_like(z)
        for m in range(2, self.cfg.max_card + 1):
            sel = card == m
            if sel.any():
                zm[sel] = z[sel] @ mats[m]
        pair = zm @ z.transpose(-1, -2)  # [B, K, W, W]; entry (a, b) = z_a^T M_m z_b
        w = z.shape[-2]
        upper = torch.ones(w, w, dtype=torch.bool, device=z.device).triu(1)
        pair_mask = mask[..., :, None] & mask[..., None, :] & upper
        return (pair * pair_mask).sum((-1, -2))
