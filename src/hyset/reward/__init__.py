"""Execution-feedback rewards rho_i = ExecScore(x_i, y_i) in [0, 1] (paper Eq. 9).

Providers:

- ``sim``: offline, deterministic simulated execution (demo/CI; no network).
- ``llm``: mock tools + a real LLM agent and LLM-as-judge through LangChain
  (``anthropic:<model>`` for Claude, ``ollama:<model>`` for a local model).
"""

from __future__ import annotations

from .base import RewardProvider, RewardRequest
from .feedback import ExecutionFeedback


def make_reward(kind: str, **kwargs) -> RewardProvider | None:
    kind = kind.lower()
    if kind in ("", "none", "off"):
        return None
    if kind in ("sim", "stub"):
        from .sim import SimulatedExecutionReward

        return SimulatedExecutionReward(**kwargs)
    if kind == "llm":
        from .llm import LangChainReward

        return LangChainReward(**kwargs)
    raise ValueError(f"unknown reward provider {kind!r} (none | sim | llm)")


__all__ = ["ExecutionFeedback", "RewardProvider", "RewardRequest", "make_reward"]
