"""Convert a ToolBench data dump (OpenBMB/ToolBench) into the hyset dataset layout.

Expected input (``root`` is ToolBench's ``data/`` directory)::

    root/instruction/G1_query.json, G2_query.json, G3_query.json
    root/test_query_ids/<split>_test_query_ids.json     (optional)
        e.g. G1_instruction, G1_tool, G1_category, G2_instruction, G2_category,
        G3_instruction; each a JSON object whose keys are query ids.

Each query record provides ``query`` (or ``instruction``), ``query_id`` (or ``id``),
``api_list`` (category_name, tool_name, api_name, api_description, ...) and
``relevant APIs`` (a list of [tool_name, api_name] pairs: the ground-truth set).

The tool library is the union of every ``api_list`` entry, in first-seen order over
G1, G2, G3. Tool ids follow ToolBench's function-name convention ``<api>_for_<tool>``.
Records with no usable ``relevant APIs`` are skipped (no fallback to ``api_list``).
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any

from ..data import Dataset, Example, Tool, ToolLibrary

log = logging.getLogger(__name__)

GROUPS = ("G1", "G2", "G3")
_RESERVED = {"from", "class", "return", "false", "true", "id", "and"}


def _norm(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", name or "")
    s = re.sub(r"_+", "_", s).strip("_").lower()
    if s and s[0].isdigit():
        s = "get_" + s
    return s


def function_name(tool_name: str, api_name: str) -> str:
    api = _norm(api_name)
    if api in _RESERVED:
        api = "is_" + api
    return f"{api}_for_{_norm(tool_name)}"


def _query_id(rec: dict[str, Any]) -> str:
    return str(rec.get("query_id", rec.get("id", "")))


def _query_text(rec: dict[str, Any]) -> str:
    q = rec.get("query", rec.get("instruction", ""))
    return " ".join(q) if isinstance(q, list) else str(q)


def _tool_from_api(api: dict[str, Any]) -> Tool:
    tool_name = api.get("tool_name", "")
    api_name = api.get("api_name", "")
    desc = (api.get("api_description") or "").strip()
    return Tool(
        id=function_name(tool_name, api_name),
        name=f"{tool_name} / {api_name}",
        description=desc or f"{api_name} endpoint of {tool_name}",
        category=api.get("category_name", ""),
    )


def convert(
    root: Path,
    *,
    val_fraction: float = 0.02,
    test_fraction: float = 0.05,
    max_train: int | None = None,
    seed: int = 0,
) -> Dataset:
    root = Path(root)
    records: list[tuple[str, dict[str, Any]]] = []
    tools: dict[str, Tool] = {}
    for g in GROUPS:
        path = root / "instruction" / f"{g}_query.json"
        if not path.exists():
            log.warning("missing %s", path)
            continue
        data = json.loads(path.read_text())
        log.info("read %d records from %s", len(data), path)
        for rec in data:
            for api in rec.get("api_list", []):
                t = _tool_from_api(api)
                tools.setdefault(t.id, t)
            records.append((g, rec))
    if not records:
        raise FileNotFoundError(f"no ToolBench query files under {root / 'instruction'}")
    library = ToolLibrary(tools.values())

    def to_example(group: str, rec: dict[str, Any], split: str) -> Example | None:
        rel = rec.get("relevant APIs", rec.get("relevant_apis")) or []
        gt: list[str] = []
        for item in rel:
            fid = function_name(item[0], item[1]) if isinstance(item, list | tuple) else str(item)
            if fid in library and fid not in gt:
                gt.append(fid)
        if not gt:
            return None
        qid = _query_id(rec)
        return Example(qid=f"{group}-{qid}", query=_query_text(rec), tools=tuple(gt),
                       meta={"group": group, "query_id": qid, "split": split})  # fmt: skip

    # official test ids, keyed per group so overlapping id namespaces don't collide
    test_ids: dict[tuple[str, str], str] = {}
    id_dir = root / "test_query_ids"
    if id_dir.exists():
        for f in sorted(id_dir.glob("*_test_query_ids.json")):
            name = f.name.removesuffix("_test_query_ids.json")
            group = name.split("_")[0]
            for qid in json.loads(f.read_text()):
                test_ids[(group, str(qid))] = name
    rng = random.Random(seed)

    train, test = [], []
    for group, rec in records:
        name = test_ids.get((group, _query_id(rec)))
        ex = to_example(group, rec, name or "train")
        if ex is None:
            continue
        (test if name else train).append(ex)
    if not test_ids:  # no official ids: hold out a random slice
        rng.shuffle(train)
        n_test = max(1, int(len(train) * test_fraction))
        test, train = train[:n_test], train[n_test:]
    rng.shuffle(train)
    n_val = max(1, int(len(train) * val_fraction))
    val, train = train[:n_val], train[n_val:]
    if max_train:
        train = train[:max_train]

    ds = Dataset(
        library,
        {"train": train, "val": val, "test": test},
        meta={"source": "toolbench", "root": str(root), "seed": seed,
              "official_test_ids": bool(test_ids), "n_tools": len(library)},
    )  # fmt: skip
    ds.validate()
    return ds
