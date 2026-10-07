from pathlib import Path

import pytest

from jev_bench.cache import DiskCache
from jev_bench.config import BenchConfig, load_config
from jev_bench.datasets import LabelSpec
from jev_bench.types import Example


@pytest.fixture(autouse=True)
def _no_real_keys(monkeypatch):
    # Tests must never reach a real API.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)


@pytest.fixture
def cfg() -> BenchConfig:
    return load_config()


@pytest.fixture
def cache(tmp_path: Path) -> DiskCache:
    return DiskCache(tmp_path / "cache")


@pytest.fixture
def spec() -> LabelSpec:
    return LabelSpec(
        instructions="Overall sentiment?",
        labels={"negative": "Dislikes it", "positive": "Likes it"},
    )


@pytest.fixture
def example(spec: LabelSpec) -> Example:
    return Example(
        id="t-0", text="A waste of two hours.", label="negative", label_space=spec.label_space
    )
