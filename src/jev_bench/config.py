"""Benchmark configuration: an editable YAML file validated by pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from jev_bench.types import Usage

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "bench.yaml"

Provider = Literal["typesafe", "anthropic"]


class BudgetConfig(BaseModel):
    cap_usd: float = Field(gt=0)
    ledger_path: Path


class LatencyConfig(BaseModel):
    calls_per_model: int = Field(gt=0)
    warmup_calls: int = Field(ge=0)
    dataset: str
    client_location: str


class SelfConsistencyConfig(BaseModel):
    model: str
    dataset: str
    n_examples: int = Field(gt=0)
    k: int = Field(ge=2)
    temperature: float = Field(gt=0)


class BootstrapConfig(BaseModel):
    n_resamples: int = Field(gt=0)


class RunConfig(BaseModel):
    seed: int
    limit: int = Field(gt=0)
    dev_limit: int = Field(gt=0)
    concurrency: dict[Provider, int]
    latency: LatencyConfig
    self_consistency: SelfConsistencyConfig
    bootstrap: BootstrapConfig
    ece_bins: int = Field(gt=0)


class ModelConfig(BaseModel):
    provider: Provider
    model_id: str
    price_input: float = Field(ge=0, description="USD per million input tokens")
    price_output: float = Field(ge=0, description="USD per million output tokens")
    batch_discount: float = Field(ge=0, lt=1)
    temperature: float | None = None
    thinking: str | None = None
    max_tokens: int = 128
    prompt_caching: bool = False

    def cost(self, usage: Usage, *, batch: bool = False) -> float:
        """USD cost of one call from real token usage.

        Cache writes bill at 1.25x input, cache reads at 0.1x (5-minute TTL).
        """
        per_in = self.price_input / 1e6
        per_out = self.price_output / 1e6
        total = (
            usage.input_tokens * per_in
            + usage.cache_creation_input_tokens * per_in * 1.25
            + usage.cache_read_input_tokens * per_in * 0.1
            + usage.output_tokens * per_out
        )
        return total * (1 - self.batch_discount) if batch else total


class DatasetConfig(BaseModel):
    hf_path: str
    hf_config: str | None = None
    split: str
    text_field: str
    label_field: str


class EstimationConfig(BaseModel):
    chars_per_token: float = Field(gt=0)
    overhead_input_tokens: dict[Provider, int]
    output_tokens: dict[Provider, int]
    tokenizer_factor: dict[str, float] = {}


class BenchConfig(BaseModel):
    budget: BudgetConfig
    cache_dir: Path
    hf_cache_dir: Path
    run: RunConfig
    estimation: EstimationConfig
    models: dict[str, ModelConfig]
    datasets: dict[str, DatasetConfig]

    def resolve(self, path: Path) -> Path:
        """Resolve a config-relative path against the repo root."""
        return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> BenchConfig:
    with path.open(encoding="utf-8-sig") as f:  # tolerate a BOM from PowerShell editors
        return BenchConfig.model_validate(yaml.safe_load(f))


def load_env() -> None:
    """Load API keys from the repo-root .env. Values are never logged."""
    load_dotenv(REPO_ROOT / ".env", override=False)
