import json
from collections import Counter

from hyset.data import Dataset
from hyset.datasets.synthetic import generate
from hyset.datasets.toolbench import convert, function_name


def test_synthetic_is_deterministic_and_valid(tmp_path):
    a = generate(n_train=200, n_val=20, n_test=30, n_filler=10, seed=7)
    b = generate(n_train=200, n_val=20, n_test=30, n_filler=10, seed=7)
    assert [e.query for e in a.splits["train"]] == [e.query for e in b.splits["train"]]
    a.save(tmp_path)
    loaded = Dataset.load(tmp_path)
    assert len(loaded.library) == len(a.library)
    assert {len(e.tools) for e in loaded.splits["train"]} <= {1, 2, 3, 4, 5}
    assert all("unseen_set" in e.meta for e in loaded.splits["test"])


def test_synthetic_cardinality_dependent_cooccurrence():
    ds = generate(n_train=3000, n_val=10, n_test=10, n_filler=0, seed=0)
    by_size = Counter()
    both = Counter()
    for e in ds.splits["train"]:
        caps = set(e.meta["capabilities"])
        by_size[len(caps)] += 1
        both[len(caps)] += {"weather_forecast", "currency_convert"} <= caps
    assert both[2] / by_size[2] < 0.03  # rare as a pair
    assert both[4] / by_size[4] > 0.2  # common in 4-tool sets


def _api(cat, tool, api, desc=""):
    return {"category_name": cat, "tool_name": tool, "api_name": api, "api_description": desc}


def test_toolbench_convert(tmp_path):
    inst = tmp_path / "instruction"
    inst.mkdir()
    g1 = [
        {"query_id": 1, "query": "weather and currency please",
         "api_list": [_api("Weather", "OpenWeather", "forecast", "daily forecast"),
                      _api("Finance", "FX Rates", "convert", "convert money")],
         "relevant APIs": [["OpenWeather", "forecast"], ["FX Rates", "convert"]]},
        {"query_id": 2, "query": ["list", "form"],
         "api_list": [_api("Data", "My-Tool", "from", "reserved word api")],
         "relevant APIs": [["My-Tool", "from"]]},
        {"query_id": 3, "query": "no labels", "api_list": [_api("X", "T", "a")]},
    ] + [
        {"query_id": 10 + i, "query": f"q{i}", "api_list": [_api("Weather", "OpenWeather", "forecast")],
         "relevant APIs": [["OpenWeather", "forecast"]]}
        for i in range(40)
    ]  # fmt: skip
    (inst / "G1_query.json").write_text(json.dumps(g1))
    ids = tmp_path / "test_query_ids"
    ids.mkdir()
    (ids / "G1_instruction_test_query_ids.json").write_text(json.dumps({"1": 0}))

    ds = convert(tmp_path, val_fraction=0.1)
    assert function_name("My-Tool", "from") == "is_from_for_my_tool"
    assert "forecast_for_openweather" in ds.library
    assert [e.qid for e in ds.splits["test"]] == ["G1-1"]
    assert set(ds.splits["test"][0].tools) == {"forecast_for_openweather", "convert_for_fx_rates"}
    all_qids = {e.qid for exs in ds.splits.values() for e in exs}
    assert "G1-3" not in all_qids  # no relevant APIs -> skipped
    assert "G1-2" in all_qids
