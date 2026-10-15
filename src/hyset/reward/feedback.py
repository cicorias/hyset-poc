"""Bookkeeping for the reward-weighted self-training loss L_self (paper Eq. 9, Alg. 1).

Only a subset of N_sub training queries receives execution feedback; the subset is
re-drawn every T_ref optimizer steps. Rewards are cached per (query, tool set), and the
total number of provider calls is capped, mirroring the paper's rollout budget.
"""

from __future__ import annotations

import logging
import random

from ..data import Example, ToolLibrary
from ..negatives import ToolSet
from .base import RewardProvider, RewardRequest

log = logging.getLogger(__name__)


class ExecutionFeedback:
    def __init__(
        self,
        provider: RewardProvider,
        examples: list[Example],
        library: ToolLibrary,
        n_sub: int = 512,
        refresh_steps: int = 500,
        max_calls: int = 20_000,
        seed: int = 0,
    ):
        self.provider = provider
        self.examples = examples
        self.library = library
        self.n_sub = min(n_sub, len(examples))
        self.refresh_steps = refresh_steps
        self.max_calls = max_calls
        self.rng = random.Random(seed)
        self.cache: dict[tuple[str, ToolSet], float] = {}
        self.calls = 0
        self.failures = 0
        self.subset: set[int] = set()

    def on_step(self, step: int) -> None:
        if step % self.refresh_steps == 0 or not self.subset:
            self.subset = set(self.rng.sample(range(len(self.examples)), self.n_sub))

    def rewards(self, batch_idx: list[int], chosen: list[ToolSet]) -> list[tuple[int, float]]:
        """For each batch row whose query is in the feedback subset, return (row, rho)
        for the model-selected set ``chosen[row]``. Unscored sets are sent to the provider
        in one call while budget remains."""
        todo: list[tuple[int, tuple[str, ToolSet]]] = []
        for row, i in enumerate(batch_idx):
            if i in self.subset:
                todo.append((row, (self.examples[i].qid, chosen[row])))

        missing = [(row, key) for row, key in todo if key not in self.cache]
        room = max(0, self.max_calls - self.calls)
        missing = missing[:room]
        if missing:
            reqs = []
            for row, (_, tool_set) in missing:
                ex = self.examples[batch_idx[row]]
                reqs.append(
                    RewardRequest(
                        qid=ex.qid,
                        query=ex.query,
                        tools=[self.library[t] for t in tool_set],
                        gt=[self.library[self.library.index(t)] for t in ex.tools],
                        meta=ex.meta,
                    )
                )
            scores = self.provider.score(reqs)
            self.calls += len(reqs)
            for (_, key), rho in zip(missing, scores, strict=True):
                if rho is None:  # failed call: leave uncached so it can be retried
                    self.failures += 1
                    continue
                self.cache[key] = float(min(1.0, max(0.0, rho)))

        return [(row, self.cache[key]) for row, key in todo if key in self.cache]
