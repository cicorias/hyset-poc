"""`hyset` command line: data, training, evaluation, retrieval and serving."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
console = Console()

DEFAULT_DATA = Path("data/synthetic")
DEFAULT_RUN = Path("runs/demo")
ROW_METRICS = ("recall@3", "recall@5", "ndcg@5", "comp@3", "comp@5")
SET_METRICS = ("set_exact", "set_cover", "set_f1", "set_size", "exec_sim")
SHORT = {"recall@3": "R@3", "recall@5": "R@5", "ndcg@5": "N@5", "comp@3": "C@3", "comp@5": "C@5",
         "set_exact": "exact", "set_cover": "cover", "set_f1": "F1", "set_size": "size",
         "exec_sim": "exec"}  # fmt: skip


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, show_time=False)],
    )
    for noisy in ("httpx", "sentence_transformers", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ------------------------------------------------------------------------- data


@app.command()
def synth(
    out: Path = typer.Argument(DEFAULT_DATA),
    n_train: int = 4000,
    n_val: int = 400,
    n_test: int = 600,
    n_filler: int = typer.Option(150, help="unrelated filler tools added to the library"),
    p_random: float = typer.Option(0.12, help="share of queries with random tool combinations"),
    p_implicit: float = typer.Option(0.35, help="chance a sub-task is phrased implicitly"),
    seed: int = 0,
) -> None:
    """Generate the synthetic tool library and labeled queries."""
    from .datasets.synthetic import generate

    ds = generate(n_train, n_val, n_test, n_filler, p_random, p_implicit, seed)
    ds.save(out)
    unseen = sum(e.meta.get("unseen_set", False) for e in ds.splits["test"])
    console.print(
        f"wrote {out}: {len(ds.library)} tools, "
        + ", ".join(f"{k}={len(v)}" for k, v in ds.splits.items())
        + f" ({unseen} test queries use tool sets unseen in train)"
    )


@app.command()
def toolbench(
    root: Path = typer.Argument(
        ..., help="ToolBench data/ dir (has instruction/, test_query_ids/)"
    ),
    out: Path = typer.Argument(Path("data/toolbench")),
    max_train: int = typer.Option(0, help="subsample training queries (0 = all)"),
    val_fraction: float = 0.02,
    seed: int = 0,
) -> None:
    """Convert a ToolBench dump into the hyset dataset layout."""
    from .datasets.toolbench import convert

    ds = convert(root, val_fraction=val_fraction, max_train=max_train or None, seed=seed)
    ds.save(out)
    console.print(
        f"wrote {out}: {len(ds.library)} tools, "
        + ", ".join(f"{k}={len(v)}" for k, v in ds.splits.items())
    )


# ------------------------------------------------------------------------- train


def _parse_mix(spec: str):
    from .negatives import NegativeMix

    if spec in ("paper", ""):
        return NegativeMix.paper()
    if spec == "mined":
        return NegativeMix.mined()
    parts = dict(kv.split("=") for kv in spec.split(","))
    return NegativeMix(**{k.strip(): float(v) for k, v in parts.items()})


@app.command()
def train(
    data: Path = typer.Argument(DEFAULT_DATA),
    out: Path = typer.Argument(DEFAULT_RUN),
    encoder: str = typer.Option("st:BAAI/bge-small-en-v1.5", help="st:<hf-id> | hashing[:dim]"),
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    z_lr_mult: float = typer.Option(1.0, help="learning-rate multiplier for tool embeddings Z"),
    k_neg: int = 64,
    neg_mix: str = typer.Option(
        "mined",
        help="'paper' (50/30/20 size_matched/in_batch/hard), 'mined' (20/20/10 + 50% mined "
        "from the model's own shortlist), or explicit "
        "'size_matched=..,in_batch=..,hard=..,perturb=..,model=..'",
    ),
    interaction: str = typer.Option("cardinality", help="cardinality | shared | identity | none"),
    max_card: int = typer.Option(5, help="M, the largest tool-set size"),
    eta: float = typer.Option(0.3, help="weight of the execution-feedback loss L_self"),
    lam: float = typer.Option(1e-2, help="weight of the interaction-matrix regularizer"),
    reward: str = typer.Option("none", help="none | sim (offline demo) | llm (LangChain)"),
    reward_mode: str = typer.Option("agent", help="llm reward: agent (rollout+judge) | judge"),
    agent_model: str = typer.Option("anthropic:claude-opus-5-5", help="init_chat_model spec"),
    judge_model: str = typer.Option("", help="defaults to --agent-model"),
    n_sub: int = typer.Option(512, help="N_sub: queries receiving feedback at a time"),
    refresh_steps: int = typer.Option(500, help="T_ref: re-draw the feedback subset every N steps"),
    max_reward_calls: int = typer.Option(20_000, help="total reward-call budget"),
    self_train_start: int = typer.Option(0, help="epoch (0-based) at which L_self kicks in"),
    k1: int = 15,
    k_pool: int = 20,
    patience: int = typer.Option(3, help="early-stopping patience, in epochs"),
    eval_every: int = typer.Option(50, help="also validate every N steps (0 = per epoch)"),
    align_scale: float = typer.Option(5.0, help="initial scale of P (attention sharpness)"),
    seed: int = 0,
) -> None:
    """Train HySet on a dataset directory and save a run directory."""
    from .data import Dataset
    from .encoders import make_encoder
    from .inference import InferenceConfig
    from .model import ModelConfig
    from .reward import make_reward
    from .run import default_cache_dir, save_run
    from .train import TrainConfig
    from .train import train as train_fn

    ds = Dataset.load(data)
    enc = make_encoder(encoder, cache_dir=default_cache_dir())
    model_cfg = ModelConfig(max_card=max_card, interaction=interaction, align_scale=align_scale,
                            seed=seed)  # type: ignore[arg-type]  # fmt: skip
    train_cfg = TrainConfig(
        epochs=epochs, batch_size=batch_size, lr=lr, z_lr_mult=z_lr_mult, k_neg=k_neg,
        mix=_parse_mix(neg_mix), eta=eta, lam=lam, patience=patience, seed=seed,
        eval_every=eval_every,
        n_sub=n_sub, refresh_steps=refresh_steps, max_reward_calls=max_reward_calls,
        self_train_start_epoch=self_train_start,
    )  # fmt: skip
    infer_cfg = InferenceConfig(k1=k1, k_pool=k_pool)
    reward_kwargs = {}
    if reward == "llm":
        reward_kwargs = {"agent_model": agent_model, "judge_model": judge_model or None,
                         "mode": reward_mode}  # fmt: skip
    provider = make_reward(reward, **reward_kwargs)
    result = train_fn(ds, enc, model_cfg, train_cfg, infer_cfg, reward=provider)
    save_run(out, result, ds.library, model_cfg=model_cfg, train_cfg=train_cfg,
             infer_cfg=infer_cfg, encoder_spec=enc.spec, dataset_dir=str(data),
             extra={"reward": reward})  # fmt: skip
    console.print(f"saved run to {out} (best epoch {result.best_epoch})")


# ------------------------------------------------------------------------- eval


def _table(title: str, rows: dict[str, dict[str, float]]) -> Table:
    table = Table(
        title=title,
        caption="R/N/C = Recall/NDCG/COMP; exact/cover/F1/size score "
        "the predicted set; exec = simulated execution reward",
    )
    table.add_column("method")
    cols = [m for m in (*ROW_METRICS, *SET_METRICS) if any(m in r for r in rows.values())]
    for m in cols:
        table.add_column(SHORT.get(m, m), justify="right")
    for name, r in rows.items():
        cells = [
            ("-" if m not in r else f"{r[m]:.2f}" if m == "set_size" else f"{100 * r[m]:.1f}")
            for m in cols
        ]
        table.add_row(name, *cells)
    return table


@app.command("eval")
def eval_cmd(
    run: Path = typer.Argument(DEFAULT_RUN),
    data: Path = typer.Option(None, help="dataset dir (default: the one used for training)"),
    split: str = "test",
    baselines: bool = True,
) -> None:
    """Evaluate a run (Recall/NDCG/COMP@K, set exact-match/cover) against baselines."""
    from .run import evaluate_run

    res = evaluate_run(run, data, split, baselines)
    console.print(_table(f"{split} ({res['n']} queries) - percentages", res["rows"]))
    if "unseen_sets" in res:
        u = res["unseen_sets"]
        console.print(_table(f"tool sets unseen in training ({u['n']} queries)", u["rows"]))
    (Path(run) / "metrics.json").write_text(json.dumps(res, indent=2))
    console.print(f"wrote {Path(run) / 'metrics.json'}")


# ------------------------------------------------------------------------- use


@app.command()
def retrieve(
    query: list[str] = typer.Argument(..., help="one or more queries"),
    run: Path = typer.Option(DEFAULT_RUN, "--run", "-r"),
    rank_k: int = 5,
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Retrieve the tool set for one or more queries."""
    from .run import Retriever

    results = Retriever.load(run).retrieve(query, rank_k=rank_k)
    if as_json:
        console.print_json(json.dumps([r.to_dict() for r in results]))
        return
    for res in results:
        console.print(f"[bold]{res.query}[/bold]")
        console.print(f"  tool set (score {res.set_score:.2f}):")
        for t in res.tools:
            console.print(f"    - {t.name}  [dim]{t.id}[/dim]")
        console.print("  ranking: " + ", ".join(t.id for t in res.ranking))


@app.command()
def serve(
    run: Path = typer.Argument(DEFAULT_RUN),
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Serve a run over HTTP (FastAPI): POST /retrieve, GET /tools, GET /health."""
    import uvicorn

    from .serve.http import create_app

    uvicorn.run(create_app(run), host=host, port=port)


@app.command()
def mcp(
    run: Path = typer.Argument(DEFAULT_RUN),
    transport: str = typer.Option("stdio", help="stdio | streamable-http | sse"),
) -> None:
    """Serve a run as an MCP server with select_tools / describe_tool."""
    from .serve.mcp_server import build_server

    logging.getLogger().handlers.clear()  # stdout belongs to the MCP protocol
    build_server(run).run(transport=transport)  # type: ignore[arg-type]


@app.command("reward-probe")
def reward_probe(
    data: Path = typer.Argument(DEFAULT_DATA),
    n: int = typer.Option(2, help="number of test queries to probe"),
    reward: str = typer.Option("llm", help="sim | llm"),
    reward_mode: str = typer.Option("agent", help="agent | judge"),
    agent_model: str = "anthropic:claude-opus-5-5",
    judge_model: str = "",
    drop_one: bool = typer.Option(True, help="also probe each GT set with one tool removed"),
) -> None:
    """Call a reward provider on a few ground-truth (and degraded) tool sets.

    A quick connectivity/sanity check for Claude or Ollama before using --reward llm."""
    from .data import Dataset
    from .reward import RewardRequest, make_reward

    ds = Dataset.load(data)
    kwargs = {}
    if reward == "llm":
        kwargs = {"agent_model": agent_model, "judge_model": judge_model or None,
                  "mode": reward_mode, "max_concurrency": 1}  # fmt: skip
    provider = make_reward(reward, **kwargs)
    lib = ds.library
    for e in ds.splits["test"][:n]:
        gt = [lib[lib.index(t)] for t in e.tools]
        variants = [("ground truth", gt)]
        if drop_one and len(gt) > 1:
            variants.append(("minus one tool", gt[:-1]))
        for label, tools in variants:
            req = RewardRequest(e.qid, e.query, tools, gt, e.meta)
            score = provider.score([req])[0]
            console.print(f"{e.qid} [{label}] reward={score}  :: {e.query}")


@app.command()
def demo(
    out: Path = typer.Option(DEFAULT_RUN, help="run directory to write"),
    data: Path = typer.Option(DEFAULT_DATA, help="synthetic dataset dir (generated if missing)"),
    encoder: str = typer.Option(
        "st:BAAI/bge-small-en-v1.5", help="use 'hashing' for a no-download demo"
    ),
    epochs: int = 10,
    reward: str = typer.Option("sim", help="none | sim | llm"),
) -> None:
    """End to end: synthesize data, train, evaluate, and retrieve the paper's example."""
    from .datasets.synthetic import PAPER_EXAMPLE
    from .run import Retriever

    if not (data / "tools.jsonl").exists():
        synth(out=data, n_train=4000, n_val=400, n_test=600, n_filler=150, p_random=0.12,
              p_implicit=0.35, seed=0)  # fmt: skip
    train(data=data, out=out, encoder=encoder, epochs=epochs, batch_size=32, lr=1e-4,
          z_lr_mult=1.0, k_neg=64, neg_mix="mined", interaction="cardinality", max_card=5,
          eta=0.3, lam=1e-2, reward=reward, reward_mode="judge",
          agent_model="anthropic:claude-opus-5-5", judge_model="", n_sub=512,
          refresh_steps=500, max_reward_calls=20_000, self_train_start=0, k1=15, k_pool=20,
          patience=3,
          eval_every=50, align_scale=5.0, seed=0)  # fmt: skip
    eval_cmd(run=out, data=data, split="test", baselines=True)
    from .baselines import dense_rankings

    console.rule("paper example (ground truth: flights, hotel, weather, currency)")
    retriever = Retriever.load(out)
    res = retriever.retrieve(PAPER_EXAMPLE)[0]
    lib, enc = retriever.library, retriever.encoder
    dense = dense_rankings(
        enc.encode(lib.texts(), "tool"), enc.encode([PAPER_EXAMPLE], "query"), 4
    )[0]
    console.print(PAPER_EXAMPLE)
    console.print("frozen dense top-4: " + ", ".join(lib[i].name for i in dense))
    console.print("HySet tool set:     " + ", ".join(t.name for t in res.tools))
