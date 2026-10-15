"""Run directories (trained checkpoints) and the high-level :class:`Retriever`.

A run directory holds everything needed to serve the model::

    <run>/model.pt      state_dict of HySet
    <run>/run.json      configs, encoder spec, dataset path, best epoch
    <run>/tools.jsonl   the tool library the model was trained on
    <run>/history.json  per-epoch training log
    <run>/metrics.json  written by `hyset eval`
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .baselines import bm25_rankings, dense_rankings
from .data import Dataset, Tool, ToolLibrary
from .encoders import Encoder, make_encoder
from .evaluate import evaluate_model, subset_rows
from .inference import InferenceConfig, SetRetriever
from .metrics import rank_metrics
from .model import HySet, ModelConfig
from .negatives import canon
from .reward.base import RewardRequest
from .reward.sim import SimulatedExecutionReward
from .train import TrainConfig, TrainResult


def default_cache_dir() -> Path:
    return Path(os.environ.get("HYSET_CACHE_DIR", ".cache/hyset"))


def save_run(
    out: Path,
    result: TrainResult,
    library: ToolLibrary,
    *,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    infer_cfg: InferenceConfig,
    encoder_spec: str,
    dataset_dir: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), out / "model.pt")
    library.save(out / "tools.jsonl")
    meta = {
        "format": 1,
        "encoder": encoder_spec,
        "d_r": result.model.d_r,
        "n_tools": len(library),
        "dataset": dataset_dir,
        "best_epoch": result.best_epoch,
        "model": asdict(model_cfg),
        "train": asdict(train_cfg),
        "inference": asdict(infer_cfg),
        **(extra or {}),
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2))
    (out / "history.json").write_text(json.dumps(result.history, indent=2))


@dataclass
class LoadedRun:
    model: HySet
    library: ToolLibrary
    meta: dict[str, Any]

    @property
    def infer_cfg(self) -> InferenceConfig:
        return InferenceConfig(**self.meta["inference"])


def load_run(run_dir: Path) -> LoadedRun:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run.json").read_text())
    library = ToolLibrary.load(run_dir / "tools.jsonl")
    model = HySet(meta["n_tools"], meta["d_r"], ModelConfig(**meta["model"]))
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    return LoadedRun(model, library, meta)


@dataclass
class RetrievalResult:
    query: str
    tools: list[Tool]  # Ê(x): the jointly scored tool set to hand to the agent
    set_score: float
    ranking: list[Tool]  # greedy set-aware ranking (for inspection / fixed budgets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "tool_set": [asdict(t) for t in self.tools],
            "set_score": self.set_score,
            "ranking": [t.id for t in self.ranking],
        }


class Retriever:
    """Query in, tool set out. Wraps a loaded run plus its frozen encoder."""

    def __init__(
        self,
        model: HySet,
        library: ToolLibrary,
        encoder: Encoder,
        infer_cfg: InferenceConfig | None = None,
    ):
        if encoder.dim != model.d_r:
            raise ValueError(f"encoder dim {encoder.dim} != model d_r {model.d_r}")
        self.model = model
        self.library = library
        self.encoder = encoder
        self.set_retriever = SetRetriever(model, infer_cfg)

    @classmethod
    def load(cls, run_dir: Path, encoder: Encoder | None = None) -> Retriever:
        run = load_run(run_dir)
        encoder = encoder or make_encoder(run.meta["encoder"], cache_dir=default_cache_dir())
        return cls(run.model, run.library, encoder, run.infer_cfg)

    def retrieve(
        self, queries: str | list[str], rank_k: int | None = None
    ) -> list[RetrievalResult]:
        if isinstance(queries, str):
            queries = [queries]
        r = torch.from_numpy(self.encoder.encode(list(queries), "query"))
        preds = self.set_retriever.predict(r, rank_k=rank_k)
        return [
            RetrievalResult(
                query=q,
                tools=[self.library[i] for i in p.tool_set],
                set_score=p.set_score,
                ranking=[self.library[i] for i in p.ranking],
            )
            for q, p in zip(queries, preds, strict=True)
        ]


def evaluate_run(
    run_dir: Path,
    data_dir: Path | None = None,
    split: str = "test",
    baselines: bool = True,
    encoder: Encoder | None = None,
    ks: tuple[int, ...] = (3, 5),
) -> dict[str, Any]:
    """Score a trained run on a dataset split, plus independent-scoring baselines.

    Rows: ``hyset`` (set-aware greedy ranking + argmax set), ``hyset-stage1`` (same
    parameters, tools ranked independently), ``dense-frozen`` (the frozen encoder's
    cosine ranking, i.e. the untrained starting point), ``bm25``."""
    run = load_run(run_dir)
    data_dir = Path(data_dir or run.meta.get("dataset") or "")
    ds = Dataset.load(data_dir)
    if [t.id for t in ds.library] != [t.id for t in run.library]:
        raise ValueError("dataset tool library differs from the one the run was trained on")
    encoder = encoder or make_encoder(run.meta["encoder"], cache_dir=default_cache_dir())
    examples = [e for e in ds.splits[split] if len(e.tools) <= run.model.cfg.max_card]
    queries = [e.query for e in examples]
    gts = [canon(ds.library.indices(e.tools)) for e in examples]
    r_np = encoder.encode(queries, "query")
    r = torch.from_numpy(r_np)

    rows, preds = evaluate_model(run.model, run.infer_cfg, r, gts, ks)
    rankings = {
        "hyset": [p.ranking for p in preds],
        "hyset-stage1": SetRetriever(run.model, run.infer_cfg).stage1_rank(r, max(ks)),
    }
    if baselines:
        tool_emb = encoder.encode(ds.library.texts(), "tool")
        rankings["dense-frozen"] = dense_rankings(tool_emb, r_np, max(ks))
        rankings["bm25"] = bm25_rankings(ds.library, queries, max(ks))
        for name in ("dense-frozen", "bm25"):
            rows[name] = rank_metrics(rankings[name], gts, ks)

    if any(t.capability for t in ds.library):
        # Offline stand-in for the paper's pass rate: mean simulated execution reward of the
        # set handed to the agent (HySet: Ê; ranking baselines: their top-5).
        sim = SimulatedExecutionReward(noise=0.0)
        handed = {"hyset": [p.tool_set for p in preds]}
        handed |= {name: [r[:5] for r in rk] for name, rk in rankings.items() if name != "hyset"}
        for name, sets in handed.items():
            reqs = [
                RewardRequest(
                    e.qid, e.query, [ds.library[i] for i in s], [ds.library[i] for i in g], e.meta
                )  # fmt: skip
                for e, s, g in zip(examples, sets, gts, strict=True)
            ]
            scores = sim.score(reqs)
            rows[name]["exec_sim"] = float(np.mean(scores))
            rows[name]["exec_pass"] = float(np.mean([s >= 0.999 for s in scores]))

    out: dict[str, Any] = {"run": str(run_dir), "dataset": str(data_dir), "split": split,
                           "n": len(examples), "rows": rows}  # fmt: skip
    unseen = np.array([bool(e.meta.get("unseen_set")) for e in examples])
    if unseen.any():
        sets = {"hyset": [p.tool_set for p in preds]}
        out["unseen_sets"] = {
            "n": int(unseen.sum()),
            "rows": subset_rows(rankings, sets, gts, unseen, ks),
        }
    return out
