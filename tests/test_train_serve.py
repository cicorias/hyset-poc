import anyio
import torch
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from hyset.cli import app as cli_app
from hyset.encoders import HashingEncoder
from hyset.inference import InferenceConfig
from hyset.model import ModelConfig
from hyset.reward.sim import SimulatedExecutionReward
from hyset.run import Retriever, evaluate_run, load_run
from hyset.serve.http import create_app
from hyset.serve.mcp_server import build_server
from hyset.train import TrainConfig, train


def test_training_beats_frozen_encoder(tiny_run, encoder):
    run_dir, data_dir, _ = tiny_run
    res = evaluate_run(run_dir, data_dir, "test", baselines=True, encoder=encoder)
    rows = res["rows"]
    assert rows["hyset"]["recall@5"] > rows["dense-frozen"]["recall@5"] + 0.1
    assert rows["hyset"]["comp@5"] > rows["dense-frozen"]["comp@5"]
    assert 1.0 <= rows["hyset"]["set_size"] <= 5.0


def test_save_load_roundtrip(tiny_run, encoder):
    run_dir, _, result = tiny_run
    loaded = load_run(run_dir)
    for a, b in zip(
        result.model.state_dict().values(), loaded.model.state_dict().values(), strict=True
    ):
        assert torch.equal(a, b)
    retr = Retriever(loaded.model, loaded.library, encoder, loaded.infer_cfg)
    out = retr.retrieve(["find flights to Paris and a hotel", "get the stock price of AAPL"])
    assert len(out) == 2 and all(1 <= len(r.tools) <= 5 for r in out)


def test_execution_feedback_path(tiny_ds, encoder):
    cfg = TrainConfig(epochs=2, lr=1e-3, k_neg=16, n_sub=64, refresh_steps=5,
                      self_train_start_epoch=0, max_reward_calls=100)  # fmt: skip
    res = train(tiny_ds, encoder, ModelConfig(), cfg, InferenceConfig(k1=8, k_pool=12),
                reward=SimulatedExecutionReward())  # fmt: skip
    assert any(h.get("reward_calls", 0) > 0 for h in res.history)
    assert max(h["reward_calls"] for h in res.history if "reward_calls" in h) <= 100
    assert any(h["l_self"] != 0 for h in res.history)


def test_http_service(tiny_run, encoder):
    run_dir, _, _ = tiny_run
    run = load_run(run_dir)
    client = TestClient(
        create_app(retriever=Retriever(run.model, run.library, encoder, run.infer_cfg))
    )
    assert client.get("/health").json()["status"] == "ok"
    body = client.post("/retrieve", json={"query": "book a hotel near the harbor"}).json()
    assert body["tool_set"] and len(body["ranking"]) == 5
    assert client.get("/tools/does-not-exist").status_code == 404
    assert len(client.post("/retrieve/batch", json={"queries": ["a", "b"]}).json()) == 2


def test_mcp_server(tiny_run, encoder):
    run_dir, _, _ = tiny_run
    run = load_run(run_dir)
    server = build_server(retriever=Retriever(run.model, run.library, encoder, run.infer_cfg))

    async def go():
        names = {t.name for t in await server.list_tools()}
        result = await server.call_tool("select_tools", {"task": "convert 100 USD to euros"})
        return names, result

    names, result = anyio.run(go)
    assert names == {"select_tools", "describe_tool"}
    assert not getattr(result, "is_error", getattr(result, "isError", False))


def test_cli_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("HYSET_CACHE_DIR", str(tmp_path / "cache"))
    runner = CliRunner()
    data, run = tmp_path / "data", tmp_path / "run"
    r = runner.invoke(cli_app, ["synth", str(data), "--n-train", "300", "--n-val", "40",
                                "--n-test", "40", "--n-filler", "10"])  # fmt: skip
    assert r.exit_code == 0, r.output
    r = runner.invoke(cli_app, ["train", str(data), str(run), "--encoder", "hashing:128",
                                "--epochs", "1", "--k-neg", "16", "--reward", "sim"])  # fmt: skip
    assert r.exit_code == 0, r.output
    r = runner.invoke(cli_app, ["eval", str(run)])
    assert r.exit_code == 0, r.output
    assert (run / "metrics.json").exists()
    r = runner.invoke(cli_app, ["retrieve", "find flights to Rome", "--run", str(run), "--json"])
    assert r.exit_code == 0 and "tool_set" in r.output


def test_encoder_dim_mismatch_is_rejected(tiny_run):
    run_dir, _, _ = tiny_run
    run = load_run(run_dir)
    try:
        Retriever(run.model, run.library, HashingEncoder(64), run.infer_cfg)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
