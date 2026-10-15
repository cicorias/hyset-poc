"""Negative-sampled candidate pools C_i = {E*_i} ∪ N_i (paper Sec. 2.4).

Negatives come from a fixed mixture of sources:

- size_matched: uniform random subsets of V with |E*| tools
- in_batch:     ground-truth sets of the other queries in the minibatch
- hard:         E* with one or two tools swapped for nearest neighbours under Z
- perturb:      E* with one tool dropped or one neighbour added.
- model:        the current model's own top-scoring sets over its inference shortlist
                (ANCE-style self-mined negatives), so training sees the near-miss sets
                that compete at inference time.

``perturb`` and ``model`` are extensions beyond the paper (weight 0 in the paper mix).

Every pool holds distinct sets, the ground truth first. Sets are sorted index tuples.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor

ToolSet = tuple[int, ...]


@dataclass
class NegativeMix:
    size_matched: float = 0.5
    in_batch: float = 0.3
    hard: float = 0.2
    perturb: float = 0.0
    model: float = 0.0

    @classmethod
    def paper(cls) -> NegativeMix:
        """The paper's fixed 50/30/20 mixture."""
        return cls()

    @classmethod
    def mined(cls) -> NegativeMix:
        """Half of the pool mined from the model's own shortlist (this PoC's default)."""
        return cls(size_matched=0.2, in_batch=0.2, hard=0.1, model=0.5)


def canon(tools) -> ToolSet:
    return tuple(sorted(set(tools)))


def _forever(fn):
    while True:
        yield fn()


class NegativeSampler:
    def __init__(
        self,
        n_tools: int,
        k_neg: int,
        max_card: int,
        mix: NegativeMix | None = None,
        n_neighbors: int = 10,
        seed: int = 0,
    ):
        if k_neg < 2:
            raise ValueError("k_neg must be >= 2 (ground truth + at least one negative)")
        self.n_tools = n_tools
        self.k_neg = k_neg
        self.max_card = max_card
        self.mix = mix or NegativeMix()
        self.n_neighbors = min(n_neighbors, n_tools - 1)
        self.rng = random.Random(seed)
        self.neighbors: list[list[int]] | None = None

        n = k_neg - 1
        self.n_in_batch = round(n * self.mix.in_batch)
        self.n_hard = round(n * self.mix.hard)
        self.n_perturb = round(n * self.mix.perturb)
        self.n_model = round(n * self.mix.model)

    @torch.no_grad()
    def refresh_neighbors(self, Z: Tensor, chunk: int = 2048) -> None:
        """Recompute each tool's nearest neighbours under the current (unit-norm) Z."""
        nbrs = []
        for start in range(0, Z.shape[0], chunk):
            sims = Z[start : start + chunk] @ Z.T
            rows = torch.arange(sims.shape[0])
            sims[rows, rows + start] = float("-inf")  # exclude self
            nbrs.append(sims.topk(self.n_neighbors, dim=-1).indices)
        self.neighbors = torch.cat(nbrs).tolist()

    # ---------------------------------------------------------------- sources

    def _random_set(self, size: int) -> ToolSet:
        return canon(self.rng.sample(range(self.n_tools), size))

    def _neighbor(self, tool: int, exclude: set[int]) -> int:
        if self.neighbors is not None:
            options = [u for u in self.neighbors[tool] if u not in exclude]
            if options:
                return self.rng.choice(options)
        while True:
            u = self.rng.randrange(self.n_tools)
            if u not in exclude:
                return u

    def _hard(self, gt: ToolSet) -> ToolSet:
        tools = list(gt)
        n_swap = 1 if len(gt) == 1 else self.rng.choice((1, 2))
        for pos in self.rng.sample(range(len(tools)), n_swap):
            tools[pos] = self._neighbor(tools[pos], set(tools))
        return canon(tools)

    def _perturb(self, gt: ToolSet) -> ToolSet:
        can_drop, can_add = len(gt) > 1, len(gt) < self.max_card
        if can_drop and (not can_add or self.rng.random() < 0.5):
            tools = list(gt)
            tools.pop(self.rng.randrange(len(tools)))
            return canon(tools)
        anchor = self.rng.choice(gt)
        return canon((*gt, self._neighbor(anchor, set(gt))))

    # ---------------------------------------------------------------- pool

    def pool(
        self, gt: ToolSet, batch_gts: list[ToolSet], mined: list[ToolSet] | None = None
    ) -> list[ToolSet]:
        """Build C_i: ``gt`` first, then distinct negatives. ``mined`` holds the model's
        top-scoring sets for this query (best first), consumed by the ``model`` source."""
        gt = canon(gt)
        seen = {gt}
        out = [gt]

        def add(cands, limit: int, attempts: int | None = None) -> None:
            added = 0
            budget = attempts if attempts is not None else 20 * max(limit, 1)
            for _, c in zip(range(budget), cands, strict=False):
                if added >= limit:
                    break
                if c not in seen and 1 <= len(c) <= self.max_card:
                    seen.add(c)
                    out.append(c)
                    added += 1

        if mined:
            add(iter(mined), self.n_model, attempts=len(mined))
        others = [canon(s) for s in batch_gts]
        self.rng.shuffle(others)
        add(iter(others), self.n_in_batch, attempts=len(others))
        add(_forever(lambda: self._hard(gt)), self.n_hard)
        add(_forever(lambda: self._perturb(gt)), self.n_perturb)
        # size-matched randoms fill the remainder (and any shortfall above)
        add(_forever(lambda: self._random_set(len(gt))), self.k_neg - len(out))
        if len(out) < self.k_neg:  # tiny libraries: fall back to any size
            sizes = range(1, min(self.max_card, self.n_tools) + 1)
            add(
                _forever(lambda: self._random_set(self.rng.choice(sizes))),
                self.k_neg - len(out),
                attempts=200 * self.k_neg,
            )
        return out


def pools_to_tensor(pools: list[list[ToolSet]], width: int) -> tuple[Tensor, Tensor]:
    """Pack pools into a padded index tensor [B, K, width] (-1 = pad) and a validity
    mask [B, K] (False where a short pool was padded with an empty row)."""
    k = max(len(p) for p in pools)
    sets = torch.full((len(pools), k, width), -1, dtype=torch.long)
    valid = torch.zeros(len(pools), k, dtype=torch.bool)
    for b, pool in enumerate(pools):
        for j, s in enumerate(pool):
            sets[b, j, : len(s)] = torch.tensor(s)
            valid[b, j] = True
    # padded rows get one dummy tool so scoring stays finite; they are masked out later
    sets[..., 0] = torch.where(valid, sets[..., 0], torch.zeros_like(sets[..., 0]))
    return sets, valid
