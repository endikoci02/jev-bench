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
    source: Literal["observed cost", "observed", "haiku-scaled", "prior"]
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


def observed_batch_cost(dataset: str, model: str) -> float | None:
    """Mean real cost per call from the accuracy (batch) log, or None.

    Unlike token counts, logged cost already reflects batch pricing and prompt-cache
    reads/writes, so it is the right basis for batch estimates. It includes one cache
    write per batch chunk, so it slightly overstates larger runs (the safe direction).
    """
    path = RAW_DIR / f"{dataset}__{model}__batch.jsonl"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        costs = [c for line in f if (c := json.loads(line)["decision"].get("cost_usd") or 0) > 0]
    return sum(costs) / len(costs) if costs else None


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
    factor = cfg.estimation.tokenizer_factor.get(model)
    reference = observed_usage(dataset, "haiku") if factor and observed is None else None
    if observed is not None:
        tin, tout, source = observed[0], observed[1], "observed"
    elif reference is not None and factor is not None:
        # Same prompt on another Claude model: scale Haiku's real usage by the
        # tokenizer factor instead of trusting the character heuristic.
        tin, tout, source = reference[0] * factor, reference[1] * factor, "haiku-scaled"
    else:
        tin, tout = prior_usage(cfg, model, spec, examples)
        source = "prior"
    m = cfg.models[model]
    per_call = (tin * m.price_input + tout * m.price_output) / 1e6
    if mode != "latency":  # accuracy and self-consistency runs go through the Batches API
        per_call *= 1 - m.batch_discount
    real = observed_batch_cost(dataset, model) if mode == "batch" else None
    if real is not None:
        per_call, source = real, "observed cost"
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
