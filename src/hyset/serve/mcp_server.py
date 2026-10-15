"""MCP server exposing HySet as a tool pre-selection step for agents.

    uv run hyset mcp runs/demo                       # stdio (for Claude Code, etc.)
    uv run hyset mcp runs/demo --transport streamable-http

An agent with a huge tool catalog calls ``select_tools`` with the user's task and loads
only the returned tool set into its context.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from ..run import Retriever

INSTRUCTIONS = """\
Tool pre-selection service. Before acting on a task that may need external tools, call \
select_tools with the task text. It returns the set of tools that are jointly useful for \
the whole task (scored as a set, not one by one), so you can load just those tools. Use \
describe_tool to read the full description of a tool by id."""


def build_server(run_dir: Path | None = None, retriever: Retriever | None = None) -> MCPServer:
    if retriever is None:
        if run_dir is None:
            raise ValueError("pass run_dir or retriever")
        retriever = Retriever.load(run_dir)
    server = MCPServer(name="hyset", instructions=INSTRUCTIONS)

    @server.tool()
    def select_tools(task: str, rank_k: int = 5) -> dict[str, Any]:
        """Select the jointly useful tool set for a task.

        Args:
            task: The user's request, in natural language.
            rank_k: How many tools to include in the backup ranking (default 5).
        Returns:
            tool_set: the tools to load (variable size, scored as a set);
            ranking: a fixed-length set-aware ranking of tool ids, for a fixed budget;
            set_score: the model's joint score for tool_set.
        """
        return retriever.retrieve(task, rank_k=max(1, min(rank_k, 50)))[0].to_dict()

    @server.tool()
    def describe_tool(tool_id: str) -> dict[str, Any]:
        """Return the full record (name, description, category) for one tool id."""
        if tool_id not in retriever.library:
            return {"error": f"unknown tool {tool_id!r}"}
        return asdict(retriever.library[retriever.library.index(tool_id)])

    return server
