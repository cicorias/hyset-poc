"""LLM-backed execution feedback through LangChain: mock tools, a real agent, an LLM judge.

Model specs follow LangChain's ``init_chat_model`` "provider:model" convention, so the
provider is a plugin choice:

- ``anthropic:claude-opus-5-5`` (default) - Claude; needs ANTHROPIC_API_KEY, honors
  ANTHROPIC_BASE_URL.
- ``ollama:qwen2.5:7b`` - a local model served by ``ollama serve`` (honors OLLAMA_HOST);
  pick one that supports tool calling for ``mode="agent"``.
- anything else ``init_chat_model`` supports, once its integration package is installed.

Modes:

- ``agent``: the agent gets the query plus only the selected tools (each a mock that
  returns a simulated payload), runs a tool-calling loop for at most ``max_steps``
  turns, and the judge grades the transcript and final answer.
- ``judge``: one judge call per request, deciding from the tool descriptions alone
  whether the set covers every sub-task. Cheaper; no agent rollout.

Rewards are continuous in [0, 1]. A request that fails (provider error, refusal,
unparseable verdict) yields ``None`` and is skipped rather than recorded as 0.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import BaseModel, Field

from ..data import Tool
from .base import RewardRequest

log = logging.getLogger(__name__)

DEFAULT_MODEL = "anthropic:claude-opus-5-5"

AGENT_SYSTEM = """\
You complete the user's request using only the tools you have been given. Call whichever \
tools you need, as many times as needed. The tools are simulated: their results are \
placeholders, so treat them as real data and do not question their content. If part of \
the request cannot be done with the tools available, say so plainly in your final answer. \
End with a concise final answer that addresses every part of the request."""

JUDGE_TRANSCRIPT_SYSTEM = """\
You grade whether a tool-using AI agent accomplished a user's request.

You will see the request, the tools the agent was allowed to use, the agent's tool calls \
with their results, and its final answer. Tool results are simulated placeholders, so do \
not grade their content; grade whether each part of the request was handled with an \
appropriate tool and reflected in the final answer.

Score on a continuous scale from 0 to 1:
- 1.0: every part of the request was handled with a suitable tool and answered.
- about 0.5: some parts were handled; others were skipped, unsupported, or made up.
- 0.0: the agent gave up, only failed, or answered a different request.
Do not reward length or effort, and do not penalize using fewer tools than were offered.

Reply with only a JSON object: {"score": <number between 0 and 1>, "reason": "<one sentence>"}"""

JUDGE_TOOLSET_SYSTEM = """\
You decide whether a set of tools is enough for an AI agent to fully complete a user's \
request. Split the request into its distinct sub-tasks and check, for each, whether at \
least one listed tool provides that capability.

Score on a continuous scale from 0 to 1: the fraction of sub-tasks that are covered, \
lowered slightly if irrelevant tools are likely to mislead the agent. Do not penalize a \
set for being small if it covers everything.

Reply with only a JSON object: {"score": <number between 0 and 1>, "reason": "<one sentence>"}"""


class _MockArgs(BaseModel):
    request: str = Field(
        description="What you want this tool to do, including any parameters "
        "(places, dates, amounts, names)."
    )


def _tool_name(tool: Tool, taken: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_-]", "_", tool.id)[:60] or "tool"
    name, i = base, 2
    while name in taken:
        name, i = f"{base}_{i}", i + 1
    taken.add(name)
    return name


def _text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def parse_score(text: str) -> float | None:
    """Pull {"score": x} out of a judge reply; tolerant of code fences and prose."""
    for match in re.finditer(r"\{.*?\}", text, flags=re.S):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "score" in obj:
            try:
                return min(1.0, max(0.0, float(obj["score"])))
            except (TypeError, ValueError):
                return None
    m = re.search(r'"?score"?\s*[:=]\s*([0-9.]+)', text)
    return min(1.0, max(0.0, float(m.group(1)))) if m else None


def _refused(message: Any) -> bool:
    meta = getattr(message, "response_metadata", {}) or {}
    return meta.get("stop_reason") == "refusal"


def init_chat(spec: str, **kwargs: Any):
    """Instantiate a LangChain chat model from "provider:model" (lazy import)."""
    try:
        from langchain.chat_models import init_chat_model
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError("LLM rewards need the 'llm' extra: uv sync --extra llm") from e
    if spec.startswith("anthropic:"):
        # Claude 5.x thinks adaptively and thinking tokens count toward max_tokens, so
        # leave room. No sampling params: current Claude models reject them.
        kwargs.setdefault("max_tokens", 8192)
    return init_chat_model(spec, **kwargs)


class LangChainReward:
    name = "llm"

    def __init__(
        self,
        agent_model: Any = DEFAULT_MODEL,
        judge_model: Any = None,
        mode: str = "agent",
        max_steps: int = 6,
        max_concurrency: int = 4,
        model_kwargs: dict[str, Any] | None = None,
    ):
        if mode not in ("agent", "judge"):
            raise ValueError("mode must be 'agent' or 'judge'")
        kwargs = model_kwargs or {}
        self.mode = mode
        self.max_steps = max_steps
        self.max_concurrency = max_concurrency
        self.agent = (
            init_chat(agent_model, **kwargs) if isinstance(agent_model, str) else agent_model
        )
        judge_model = judge_model if judge_model is not None else agent_model
        if judge_model is agent_model and not isinstance(agent_model, str):
            self.judge = self.agent
        else:
            self.judge = (
                init_chat(judge_model, **kwargs) if isinstance(judge_model, str) else judge_model
            )

    # ------------------------------------------------------------------ agent

    def _mock_tools(self, tools: list[Tool]):
        from langchain_core.tools import StructuredTool

        taken: set[str] = set()
        out = []
        for t in tools:

            def run(request: str, _t: Tool = t) -> str:
                return json.dumps(
                    {
                        "tool": _t.name,
                        "status": "ok",
                        "simulated": True,
                        "result": f"Simulated {_t.capability or _t.name} result for: {request}",
                    }
                )

            out.append(
                StructuredTool.from_function(
                    func=run,
                    name=_tool_name(t, taken),
                    description=f"{t.name}: {t.description}"[:1000],
                    args_schema=_MockArgs,
                )
            )
        return out

    def run_agent(self, req: RewardRequest) -> tuple[str, str]:
        """Run the tool-calling loop; return (transcript, final_answer)."""
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        tools = self._mock_tools(req.tools)
        by_name = {t.name: t for t in tools}
        llm = self.agent.bind_tools(tools) if tools else self.agent
        messages: list[Any] = [SystemMessage(AGENT_SYSTEM), HumanMessage(req.query)]
        steps: list[str] = []
        answer = ""
        for _ in range(self.max_steps):
            ai = llm.invoke(messages)
            if _refused(ai):
                raise RuntimeError("agent model refused the request")
            messages.append(ai)  # append unchanged: keeps any thinking blocks valid
            calls = getattr(ai, "tool_calls", None) or []
            if not calls:
                answer = _text(ai)
                break
            for call in calls:
                tool = by_name.get(call["name"])
                if tool is None:
                    result = json.dumps({"error": f"unknown tool {call['name']}"})
                else:
                    result = tool.invoke(call.get("args", {}))
                steps.append(f"CALL {call['name']}({json.dumps(call.get('args', {}))}) -> {result}")
                messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
        else:
            steps.append(f"(stopped after {self.max_steps} steps without a final answer)")
        return "\n".join(steps) or "(no tool calls)", answer

    # ------------------------------------------------------------------ judge

    @staticmethod
    def _tool_list(tools: list[Tool]) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in tools) or "(none)"

    def _judge(self, system: str, user: str) -> float | None:
        from langchain_core.messages import HumanMessage, SystemMessage

        reply = self.judge.invoke([SystemMessage(system), HumanMessage(user)])
        if _refused(reply):
            log.warning("judge refused; skipping")
            return None
        score = parse_score(_text(reply))
        if score is None:
            log.warning("could not parse judge reply: %.200s", _text(reply))
        return score

    def _score_one(self, req: RewardRequest) -> float | None:
        try:
            if self.mode == "judge":
                user = f"Request:\n{req.query}\n\nTools:\n{self._tool_list(req.tools)}"
                return self._judge(JUDGE_TOOLSET_SYSTEM, user)
            transcript, answer = self.run_agent(req)
            user = (
                f"Request:\n{req.query}\n\nTools offered:\n{self._tool_list(req.tools)}\n\n"
                f"Tool calls:\n{transcript}\n\nFinal answer:\n{answer or '(none)'}"
            )
            return self._judge(JUDGE_TRANSCRIPT_SYSTEM, user)
        except Exception as e:  # provider/network errors must not crash training
            log.warning("reward request %s failed: %s", req.qid, e)
            return None

    def score(self, requests: list[RewardRequest]) -> list[float | None]:
        if len(requests) <= 1 or self.max_concurrency <= 1:
            return [self._score_one(r) for r in requests]
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            return list(pool.map(self._score_one, requests))
