from pathlib import Path

import pytest

from jev_bench.config import DEFAULT_CONFIG_PATH, load_config
from jev_bench.types import Usage


def test_default_config_loads():
    cfg = load_config()
    assert cfg.budget.cap_usd == 4.00
    assert set(cfg.models) == {"jev", "haiku", "sonnet"}
    assert cfg.models["sonnet"].temperature is None
    assert cfg.models["haiku"].temperature == 0.0
    assert set(cfg.datasets) == {"sst2", "banking77", "clinc150", "offensive"}


def test_config_tolerates_utf8_bom(tmp_path: Path):
    p = tmp_path / "bench.yaml"
    p.write_bytes(b"\xef\xbb\xbf" + DEFAULT_CONFIG_PATH.read_bytes())
    assert load_config(p).run.seed == 42


def test_cost_standard_and_batch():
    haiku = load_config().models["haiku"]
    u = Usage(input_tokens=1_000_000, output_tokens=100_000)
    assert haiku.cost(u) == pytest.approx(1.0 + 0.5)
    assert haiku.cost(u, batch=True) == pytest.approx(0.75)


def test_cost_with_cache_tokens():
    sonnet = load_config().models["sonnet"]
    u = Usage(
        input_tokens=0, cache_creation_input_tokens=1_000_000, cache_read_input_tokens=1_000_000
    )
    assert sonnet.cost(u) == pytest.approx(2.0 * 1.25 + 2.0 * 0.1)


def test_jev_output_tokens_free():
    jev = load_config().models["jev"]
    assert jev.cost(Usage(input_tokens=320, output_tokens=32)) == pytest.approx(320 * 0.042 / 1e6)
