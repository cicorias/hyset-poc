from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..data import Tool


@dataclass
class RewardRequest:
    qid: str
    query: str
    tools: list[Tool]  # the model-selected set Ê_i the agent is restricted to
    gt: list[Tool] = field(default_factory=list)  # annotated E*_i (only `sim` reads it)
    meta: dict[str, Any] = field(default_factory=dict)


class RewardProvider(Protocol):
    name: str

    def score(self, requests: list[RewardRequest]) -> list[float | None]:
        """Return one reward in [0, 1] per request, in order; None = no verdict (skip)."""
        ...
