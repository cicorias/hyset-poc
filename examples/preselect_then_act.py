"""HySet as a pre-selection module in front of an unmodified agent.

HySet picks the tool set for the query; a LangChain tool-calling agent then sees only
those tools (as mocks that return simulated payloads). Costs tokens when the agent is
Claude.

    uv run python examples/preselect_then_act.py \
        "Find flights from Boston to Rome next month, book a hotel near the Colosseum, \
         and tell me what to pack" \
        --run runs/demo --model anthropic:claude-opus-5-5     # or ollama:qwen2.5:7b
"""

from __future__ import annotations

import argparse

from hyset.reward import RewardRequest
from hyset.reward.llm import LangChainReward
from hyset.run import Retriever


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("query")
    ap.add_argument("--run", default="runs/demo")
    ap.add_argument("--model", default="anthropic:claude-opus-5-5")
    ap.add_argument("--max-steps", type=int, default=6)
    args = ap.parse_args()

    selection = Retriever.load(args.run).retrieve(args.query)[0]
    print("HySet selected:", ", ".join(t.name for t in selection.tools))

    agent = LangChainReward(agent_model=args.model, mode="agent", max_steps=args.max_steps)
    transcript, answer = agent.run_agent(RewardRequest("example", args.query, selection.tools))
    print("\n--- tool calls ---\n" + transcript)
    print("\n--- final answer ---\n" + (answer or "(none)"))


if __name__ == "__main__":
    main()
