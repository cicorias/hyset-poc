"""Offline simulated execution reward (demo mode; no network, deterministic).

Pretends a competent agent ran the query with only the selected tools:

- it succeeds on a sub-task iff some selected tool provides the needed capability
  (tools without a capability label are matched by id instead), so a functionally
  equivalent tool earns credit even though it is not the annotated one;
- reward = (fraction of required capabilities covered) ** sharpness
  - distraction_penalty * (number of selected tools that cover nothing);
- with probability ``noise`` the outcome is flipped, emulating judge noise. The flip is
  seeded by (qid, tool set) so repeated calls agree.
"""

from __future__ import annotations

import hashlib
import random

from ..data import Tool
from .base import RewardRequest


def _cap(t: Tool) -> str:
    return t.capability or f"id:{t.id}"


class SimulatedExecutionReward:
    name = "sim"

    def __init__(
        self,
        sharpness: float = 2.0,
        distraction_penalty: float = 0.05,
        noise: float = 0.05,
        seed: int = 0,
    ):
        self.sharpness = sharpness
        self.distraction_penalty = distraction_penalty
        self.noise = noise
        self.seed = seed

    def _one(self, req: RewardRequest) -> float:
        required = {_cap(t) for t in req.gt}
        if not required and "capabilities" in req.meta:
            required = set(req.meta["capabilities"])
        provided = [_cap(t) for t in req.tools]
        covered = required & set(provided)
        useless = sum(1 for c in provided if c not in required)
        rho = (len(covered) / len(required)) ** self.sharpness if required else 0.0
        rho = max(0.0, rho - self.distraction_penalty * useless)
        if self.noise:
            key = f"{self.seed}|{req.qid}|{sorted(t.id for t in req.tools)}"
            h = int.from_bytes(hashlib.sha1(key.encode()).digest()[:8], "big")
            if random.Random(h).random() < self.noise:
                rho = 1.0 - rho
        return rho

    def score(self, requests: list[RewardRequest]) -> list[float]:
        return [self._one(r) for r in requests]
