"""HTTP service: POST a query, get back the tool set to hand to your agent.

    uv run hyset serve runs/demo --port 8000
    curl -s localhost:8000/retrieve -H 'content-type: application/json' \\
         -d '{"query": "Find flights to Tokyo and check the weather"}'
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..run import Retriever


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    rank_k: int = Field(default=5, ge=1, le=50, description="length of the greedy ranking")


class RetrieveBatchRequest(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=256)
    rank_k: int = Field(default=5, ge=1, le=50)


def create_app(run_dir: Path | None = None, retriever: Retriever | None = None) -> FastAPI:
    if retriever is None:
        if run_dir is None:
            raise ValueError("pass run_dir or retriever")
        retriever = Retriever.load(run_dir)
    app = FastAPI(
        title="HySet tool retrieval",
        description="Set-level tool pre-selection for LLM agents (arXiv:2607.25718).",
    )

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "n_tools": len(retriever.library),
            "encoder": retriever.encoder.spec,
            "max_card": retriever.model.cfg.max_card,
        }

    @app.get("/tools")
    def tools() -> list[dict]:
        return [asdict(t) for t in retriever.library]

    @app.get("/tools/{tool_id}")
    def tool(tool_id: str) -> dict:
        if tool_id not in retriever.library:
            raise HTTPException(404, f"unknown tool {tool_id!r}")
        return asdict(retriever.library[retriever.library.index(tool_id)])

    @app.post("/retrieve")
    def retrieve(req: RetrieveRequest) -> dict:
        return retriever.retrieve(req.query, rank_k=req.rank_k)[0].to_dict()

    @app.post("/retrieve/batch")
    def retrieve_batch(req: RetrieveBatchRequest) -> list[dict]:
        return [r.to_dict() for r in retriever.retrieve(req.queries, rank_k=req.rank_k)]

    return app
