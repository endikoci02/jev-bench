import json

import pytest

from jev_bench.report import load_units, write_report
from jev_bench.types import Decision, Usage


def _write(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _rec(model, mode, i, gold, d, warmup=False):
    return {
        "run_id": "r",
        "dataset": "sst2",
        "model": model,
        "mode": mode,
        "example_id": f"e{i}",
        "gold": gold,
        "warmup": warmup,
        "decision": d.model_dump(mode="json"),
    }


@pytest.fixture
def raw(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    golds = ["negative", "positive"] * 5
    jev = [
        Decision(
            label=g if i != 2 else "positive",
            confidence=1.0,
            api_confidence=1.0,
            confidence_type="native_max_prob",
            probabilities={"negative": 1.0},
            usage=Usage(input_tokens=350, output_tokens=31),
            model_version="jev-1.13.0",
        )
        for i, g in enumerate(golds)
    ]
    haiku = [
        Decision(
            label=g,
            confidence=0.85,
            confidence_type="verbalized",
            usage=Usage(input_tokens=300, output_tokens=17),
        )
        for g in golds[:-1]
    ]
    haiku.append(
        Decision(label=None, confidence=None, confidence_type="verbalized", error="refusal")
    )
    lat = [
        Decision(
            label="negative",
            confidence=1.0,
            confidence_type="native_max_prob",
            latency_ms=1000.0 if i == 0 else 100.0 + i,
        )
        for i in range(6)
    ]
    _write(
        raw / "sst2__jev__batch.jsonl",
        [_rec("jev", "batch", i, g, d) for i, (g, d) in enumerate(zip(golds, jev, strict=True))],
    )
    _write(
        raw / "sst2__haiku__batch.jsonl",
        [
            _rec("haiku", "batch", i, g, d)
            for i, (g, d) in enumerate(zip(golds, haiku, strict=True))
        ],
    )
    _write(
        raw / "sst2__jev__latency.jsonl",
        [_rec("jev", "latency", i, "negative", d, warmup=i == 0) for i, d in enumerate(lat)],
    )
    return raw


def test_units_include_jev_api_confidence_variant(raw):
    methods = {(u.model, u.mode, u.method) for u in load_units(raw)}
    assert ("jev", "batch", "native max-prob") in methods
    assert ("jev", "batch", "API confidence") in methods
    assert ("haiku", "batch", "verbalized") in methods


def test_write_report_end_to_end(cfg, raw, tmp_path):
    out = write_report(cfg, raw_dir=raw, out_dir=tmp_path / "results")
    import pandas as pd

    df = pd.read_csv(tmp_path / "results" / "summary.csv")
    jev = df[(df.model == "jev") & (df.confidence_method == "native max-prob")].iloc[0]
    assert jev.accuracy == pytest.approx(0.9)
    assert jev.hard_share == 1.0 and jev.ece == pytest.approx(0.1)
    assert jev["confident_wrong_0.9"] == pytest.approx(0.1)
    assert jev.latency_p50_ms == pytest.approx(103.0)  # warmup (1000 ms) excluded
    haiku = df[df.model == "haiku"].iloc[0]
    assert haiku.accuracy == pytest.approx(0.9)  # refusal scored wrong
    assert haiku.answered == 9 and haiku.failure_rate == pytest.approx(0.1)
    assert haiku.cost_per_1k_batch == pytest.approx(haiku.cost_per_1k_standard / 2)
    md = (tmp_path / "results" / "summary.md").read_text(encoding="utf-8")
    assert "Haiku 4.5 (verbalized)" in md and "Latency" in md
    names = {p.name for p in out["charts"]}
    assert {
        "reliability_sst2.png",
        "selective_sst2.png",
        "accuracy_vs_latency.png",
        "accuracy_vs_cost.png",
    } <= names
    assert all(p.stat().st_size > 10_000 for p in out["charts"])
