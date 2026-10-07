"""Build the right Decider for a configured model name."""

from __future__ import annotations

from jev_bench.adapters.claude import ClaudeDecider
from jev_bench.adapters.jev import JevDecider
from jev_bench.cache import DiskCache
from jev_bench.config import BenchConfig
from jev_bench.datasets import LabelSpec


def make_decider(
    cfg: BenchConfig, model: str, spec: LabelSpec, cache: DiskCache
) -> JevDecider | ClaudeDecider:
    m = cfg.models[model]
    if m.provider == "typesafe":
        return JevDecider(model, m, spec, cache)
    return ClaudeDecider(model, m, spec, cache)
