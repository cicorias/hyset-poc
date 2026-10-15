"""Core data types: tools, labeled queries, and on-disk dataset layout.

A dataset directory looks like::

    <dir>/tools.jsonl     one Tool per line
    <dir>/train.jsonl     one Example per line (likewise val.jsonl, test.jsonl)
    <dir>/meta.json       free-form provenance (generator, seed, source paths)
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Tool:
    """One node of the tool co-invocation hypergraph (a callable API endpoint)."""

    id: str
    name: str
    description: str
    category: str = ""
    # Functional capability label. Optional; used by the synthetic data and the
    # simulated reward to decide whether a tool set can actually solve a query.
    capability: str = ""

    def text(self) -> str:
        """Text fed to the frozen tool encoder to initialize this tool's embedding."""
        parts = [self.name]
        if self.category:
            parts.append(f"({self.category})")
        return f"{' '.join(parts)}: {self.description}"


@dataclass
class Example:
    """A query x_i with its annotated, unordered ground-truth tool set E*_i."""

    qid: str
    query: str
    tools: tuple[str, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"qid": self.qid, "query": self.query, "tools": list(self.tools), "meta": self.meta}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Example:
        return cls(
            qid=str(d["qid"]),
            query=d["query"],
            tools=tuple(d["tools"]),
            meta=d.get("meta", {}),
        )


class ToolLibrary:
    """The tool node set V with a stable id <-> index mapping."""

    def __init__(self, tools: Iterable[Tool]):
        self.tools: list[Tool] = list(tools)
        self._index = {t.id: i for i, t in enumerate(self.tools)}
        if len(self._index) != len(self.tools):
            raise ValueError("duplicate tool ids in library")

    def __len__(self) -> int:
        return len(self.tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self.tools)

    def __getitem__(self, i: int) -> Tool:
        return self.tools[i]

    def __contains__(self, tool_id: object) -> bool:
        return tool_id in self._index

    def index(self, tool_id: str) -> int:
        return self._index[tool_id]

    def indices(self, tool_ids: Iterable[str]) -> tuple[int, ...]:
        return tuple(self._index[t] for t in tool_ids)

    def texts(self) -> list[str]:
        return [t.text() for t in self.tools]

    def save(self, path: Path) -> None:
        write_jsonl(path, (asdict(t) for t in self.tools))

    @classmethod
    def load(cls, path: Path) -> ToolLibrary:
        return cls(Tool(**d) for d in read_jsonl(path))


@dataclass
class Dataset:
    library: ToolLibrary
    splits: dict[str, list[Example]]
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.library.save(root / "tools.jsonl")
        for name, examples in self.splits.items():
            write_jsonl(root / f"{name}.jsonl", (e.to_json() for e in examples))
        (root / "meta.json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, root: Path) -> Dataset:
        root = Path(root)
        library = ToolLibrary.load(root / "tools.jsonl")
        splits = {
            name: [Example.from_json(d) for d in read_jsonl(root / f"{name}.jsonl")]
            for name in SPLITS
            if (root / f"{name}.jsonl").exists()
        }
        meta_path = root / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        ds = cls(library, splits, meta)
        ds.validate()
        return ds

    def validate(self) -> None:
        for name, examples in self.splits.items():
            for e in examples:
                missing = [t for t in e.tools if t not in self.library]
                if missing:
                    raise ValueError(f"{name}/{e.qid}: unknown tool ids {missing}")
                if not e.tools:
                    raise ValueError(f"{name}/{e.qid}: empty ground-truth tool set")

    def max_cardinality(self) -> int:
        return max(len(e.tools) for exs in self.splits.values() for e in exs)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
