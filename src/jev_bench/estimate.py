"""Pre-run call counts and cost estimates.

Token counts come from logged real usage when available (results/raw); otherwise
from the priors in config. Calls already in the disk cache are free and excluded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from jev_bench.config import REPO_ROOT, BenchConfig
from jev_bench.datasets import LabelSpec
from jev_bench.prompts import system_prompt
from jev_bench.types import Example

Mode = Literal["batch", "latency", "self_consistency"]
RAW_DIR = REPO_ROOT / "results" / "raw"


def raw_log_path(dataset: str, model: str, mode: Mode) -> Path:
    return RAW_DIR / f"{dataset}__{model}__{mode}.jsonl"


@dataclass(frozen=True)
class EstimateRow:
    dataset: str
    model: str
    mode: Mode
    calls: int
    cached_calls: int
    input_tokens_per_call: float
    output_tokens_per_call: float
    source: Literal["observed", "prior"]
    cost_usd: float

    @property
    def paid(self) -> bool:
        return self.model != "jev"


def observed_usage(dataset: str, model: str) -> tuple[float, float] | None:
    """Mean (input incl. cache tokens, output) per uncached call across logged runs."""
    ins: list[int] = []
    outs: list[int] = []
    for path in RAW_DIR.glob(f"{dataset}__{model}__*.jsonl"):
        if path.stem.endswith("__self_consistency"):  # usage there is summed over k samples
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)["decision"]
                u = d.get("usage") or {}
                total_in = (
                    u.get("input_tokens", 0)
                    + u.get("cache_creation_input_tokens", 0)
                    + u.get("cache_read_input_tokens", 0)
                )
                if total_in:
                    ins.append(total_in)
                    outs.append(u.get("output_tokens", 0))
    if not ins:
        return None
    return sum(ins) / len(ins), sum(outs) / len(outs)


def prior_usage(
    cfg: BenchConfig, model: str, spec: LabelSpec, examples: list[Example]
) -> tuple[float, float]:
    m = cfg.models[model]
    est = cfg.estimation
    mean_text = sum(len(e.text) for e in examples) / max(1, len(examples))
    if m.provider == "typesafe":
        prompt_chars = len(spec.instructions) + sum(len(k) + len(v) for k, v in spec.labels.items())
    else:
        prompt_chars = len(system_prompt(spec)) + sum(len(k) + 8 for k in spec.labels)  # + enum
    factor = est.tokenizer_factor.get(model, 1.0)
    tokens_in = (
        (prompt_chars + mean_text) / est.chars_per_token + est.overhead_input_tokens[m.provider]
    ) * factor
    return tokens_in, est.output_tokens[m.provider] * factor


def estimate_row(
    cfg: BenchConfig,
    *,
    dataset: str,
    model: str,
    mode: Mode,
    spec: LabelSpec,
    examples: list[Example],
    calls: int,
    cached_calls: int,
) -> EstimateRow:
    observed = observed_usage(dataset, model)
    if observed is not None:
        tin, tout, source = observed[0], observed[1], "observed"
    else:
        tin, tout = prior_usage(cfg, model, spec, examples)
        source = "prior"
    m = cfg.models[model]
    per_call = (tin * m.price_input + tout * m.price_output) / 1e6
    if mode != "latency":  # accuracy and self-consistency runs go through the Batches API
        per_call *= 1 - m.batch_discount
    return EstimateRow(
        dataset=dataset,
        model=model,
        mode=mode,
        calls=calls,
        cached_calls=cached_calls,
        input_tokens_per_call=tin,
        output_tokens_per_call=tout,
        source=source,  # type: ignore[arg-type]
        cost_usd=per_call * (calls - cached_calls),
    )
