# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, Copilot, …) working in this repo.
`CLAUDE.md` imports this file; keep shared instructions here.

## What this is

A clean-room proof of concept of **HySet** — *"Tools Are Not Islands: Set-Level Tool
Retrieval for LLM Agents via Query-Conditioned Hyperedge Prediction"* (arXiv:2607.25718;
download the PDF to `docs/2607.25718.pdf` — it is git-ignored, not redistributed). Instead of ranking tools one by one, HySet scores whole
candidate tool **sets** (hyperedges) for a query and hands the best set to a frozen
downstream agent.

**Clean room.** The authors' code (github.com/stormwther18/HYSET, MIT) was used only
through a written functional spec, never read or copied. Do not paste code from that
repo or its HF Space into this one; implement from the paper and `docs/DESIGN.md`.

## Toolchain rules (always)

- **mise** pins the toolchain (`mise.toml`: Python 3.12 + uv) and runs tasks.
- **uv** manages dependencies into the project **`.venv`**; never `pip install`, never
  use system Python, never create other venvs.
- Add deps with `uv add <pkg>` (core), `uv add --optional llm <pkg>` (extras), or
  `uv add --group dev <pkg>`; commit `pyproject.toml` + `uv.lock` together.
- Run code with `uv run …` or `mise run <task>`. `mise.toml` forces uv to build the venv
  from mise's interpreter (`UV_PYTHON_PREFERENCE=only-system`).
- torch comes from the **CPU-only** PyTorch index (`[tool.uv.sources]`). On a GPU box,
  remove that source/index and re-lock.

```bash
mise install          # python + uv
mise run setup        # uv sync --all-extras  (creates .venv)
mise run test         # fast suite (~30 s, no downloads, no network)
mise run lint         # ruff check + format --check
mise run fmt          # auto-fix
mise run demo         # synth data -> train -> eval -> retrieve (downloads bge-small once)
```

## Layout

```
src/hyset/
  model.py         HySet nn.Module: F = F_set + F_align, cardinality-specific M_m
  inference.py     two-stage inference: shortlist + exhaustive set argmax, greedy ranking
  negatives.py     negative pools (size-matched / in-batch / hard / perturb / model-mined)
  train.py         training loop: L_ret + eta*L_self + lambda*||M||^2, early stopping
  evaluate.py      model-level metrics rows;  metrics.py  Recall/NDCG/COMP@K, set metrics
  baselines.py     BM25 + frozen dense retriever (independent scoring)
  encoders.py      frozen encoders: sentence-transformers (default bge-small) | hashing
  data.py          Tool / Example / ToolLibrary / Dataset (JSONL on disk)
  run.py           run dirs (save/load), Retriever (query -> tool set), evaluate_run
  datasets/        synthetic.py (planted set structure), toolbench.py (converter)
  reward/          execution feedback: sim.py (offline), llm.py (LangChain: Claude/Ollama),
                   feedback.py (N_sub subset, T_ref refresh, cache, call budget)
  serve/           http.py (FastAPI), mcp_server.py (MCP v2 MCPServer)
  cli.py           `hyset` Typer CLI
tests/             pytest; hashing encoder only, so no downloads
docs/              paper PDF, DESIGN.md (paper->code map, deviations, findings)
data/ runs/ .cache/  generated, git-ignored
```

## Paper → code (quick map; details in `docs/DESIGN.md`)

| Paper | Code |
|---|---|
| Eq. 2 F = F_set + F_align | `HySet.score_sets` |
| Eq. 3 F_set = Σ z_aᵀ M_m z_b | `HySet._f_set`, `HySet.interaction_matrices` |
| Eq. 4–6 attention-pooled alignment | `HySet.score_sets` (alpha / ell) |
| Eq. 8 L_ret, Eq. 9 L_self, Eq. 10 reg | `train.train` |
| Eq. 11 ‖z_j‖ = 1 | `HySet.Z` (normalized in forward) |
| Eq. 12–13 shortlist + complementarity expansion | `SetRetriever.shortlist` |
| Eq. 14 exhaustive argmax over 1..M subsets | `SetRetriever.best_sets` |
| Greedy marginal-gain ranking | `SetRetriever.greedy_rank` |
| Negative mix 50/30/20 | `NegativeMix.paper()` |
| Theorem 1 (pairwise-decomposable iff M_2=…=M_M) | `tests/test_model.py::test_theorem1_*` |

## Conventions

- Tool sets are **sorted index tuples** (`negatives.canon`); padded tensors use `-1`.
- Model code takes precomputed frozen query embeddings `r` (`[B, d_r]`); encoders live
  outside the module. `d_z` defaults to the encoder width.
- Configs are dataclasses (`ModelConfig`, `TrainConfig`, `InferenceConfig`,
  `NegativeMix`) serialized into `run.json`; add new fields **with defaults** so old runs
  still load.
- Keep `score_sets` and the fast enumerator in `inference.py` numerically identical —
  tests compare them against brute force. If you change F, update both.
- Logging via `logging.getLogger(__name__)`; the CLI installs a Rich handler.
- Style: ruff (line length 100). Comments explain *why* and cite paper equations.

## Testing

- `mise run test` must stay fast and offline: use `HashingEncoder`, small synthetic
  data, few epochs. Mark tests that download models `@pytest.mark.slow` and tests that
  call a real LLM `@pytest.mark.llm` (both deselected by default).
- New inference/scoring code needs a brute-force equivalence test.
- LLM reward code is tested with LangChain fake chat models (`tests/test_reward.py`).

## Gotchas

- **Never run `--reward llm` or `hyset reward-probe` against Claude without the user's
  OK** — every rollout/judge call costs money. `--reward sim` is the free offline stand-in.
- Claude via LangChain: specs look like `anthropic:claude-opus-5-5`. Current Claude
  models reject sampling params (`temperature`, `top_p`) and forced `tool_choice`; don't
  add them. Thinking tokens count toward `max_tokens` (we default to 8192).
  `ANTHROPIC_API_KEY` is required; `ANTHROPIC_BASE_URL` is honored.
- Ollama: `ollama:<model>` needs `ollama serve` running; agent mode needs a model with
  tool calling.
- MCP SDK is **v2**: `from mcp.server.mcpserver import MCPServer` (FastMCP was renamed).
- Embedding cache: `.cache/hyset/` (override `HYSET_CACHE_DIR`), keyed by encoder spec +
  exact texts. Delete it if you change how an encoder embeds text.
- `hyset mcp` speaks on stdout — never print/log to stdout in code paths it uses.

## Extending

- **Encoder**: implement `encoders.Encoder` (`spec`, `dim`, `encode(texts, kind)`
  returning L2-normalized float32) and route a spec prefix in `make_encoder`.
- **Reward provider**: implement `reward.base.RewardProvider.score(requests) ->
  list[float | None]` (None = no verdict, skipped) and register it in `make_reward`.
- **Dataset**: write a converter to `Dataset(library, {"train","val","test"})` and call
  `Dataset.save(dir)`; every ground-truth tool id must exist in the library.
