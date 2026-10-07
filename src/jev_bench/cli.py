"""Command-line interface: `jev-bench estimate | run | report | budget`."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from jev_bench.adapters.claude import ClaudeDecider
from jev_bench.budget import SpendLedger
from jev_bench.cache import DiskCache
from jev_bench.config import BenchConfig, load_config
from jev_bench.datasets import load_examples, load_label_spec
from jev_bench.estimate import EstimateRow, Mode, estimate_row
from jev_bench.registry import make_decider

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

ALL_MODES: tuple[Mode, ...] = ("batch", "latency", "self_consistency")


def _split(value: str | None, default: list[str]) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()] if value else default


def _ledger(cfg: BenchConfig) -> SpendLedger:
    return SpendLedger(cfg.resolve(cfg.budget.ledger_path), cfg.budget.cap_usd)


def _cached_count(cfg: BenchConfig, model: str, dataset: str, examples: list, k: int = 1) -> int:
    spec = load_label_spec(dataset)
    cache = DiskCache(cfg.resolve(cfg.cache_dir))
    decider = make_decider(cfg, model, spec, cache)
    n = 0
    for ex in examples:
        for i in range(k):
            if isinstance(decider, ClaudeDecider):
                temp = cfg.run.self_consistency.temperature if k > 1 else None
                key, _ = decider.batch_request(ex, temperature=temp, sample_index=i)
            else:
                key = decider.key(ex)
            n += key in cache
    return n


def build_estimates(
    cfg: BenchConfig, datasets: list[str], models: list[str], limit: int, modes: list[Mode]
) -> list[EstimateRow]:
    rows: list[EstimateRow] = []
    if "batch" in modes:
        for ds in datasets:
            spec = load_label_spec(ds)
            examples = load_examples(ds, cfg, limit)
            for m in models:
                rows.append(
                    estimate_row(
                        cfg, dataset=ds, model=m, mode="batch", spec=spec, examples=examples,
                        calls=len(examples),
                        cached_calls=_cached_count(cfg, m, ds, examples),
                    )
                )  # fmt: skip
    if "latency" in modes:
        lat = cfg.run.latency
        spec = load_label_spec(lat.dataset)
        examples = load_examples(lat.dataset, cfg, lat.calls_per_model + lat.warmup_calls)
        for m in models:
            rows.append(
                estimate_row(
                    cfg, dataset=lat.dataset, model=m, mode="latency", spec=spec,
                    examples=examples, calls=len(examples), cached_calls=0,
                )
            )  # fmt: skip
    if "self_consistency" in modes:
        sc = cfg.run.self_consistency
        if sc.model in models:
            spec = load_label_spec(sc.dataset)
            examples = load_examples(sc.dataset, cfg, sc.n_examples)
            rows.append(
                estimate_row(
                    cfg, dataset=sc.dataset, model=sc.model, mode="self_consistency", spec=spec,
                    examples=examples, calls=len(examples) * sc.k,
                    cached_calls=_cached_count(cfg, sc.model, sc.dataset, examples, k=sc.k),
                )
            )  # fmt: skip
    return rows


@app.command()
def estimate(
    datasets: Annotated[str | None, typer.Option(help="Comma-separated; default all")] = None,
    models: Annotated[str | None, typer.Option(help="Comma-separated; default all")] = None,
    limit: Annotated[int | None, typer.Option(help="Examples per dataset")] = None,
    mode: Annotated[str, typer.Option(help="batch, latency, self_consistency or all")] = "all",
) -> None:
    """Show call counts and estimated cost per model before anything runs."""
    cfg = load_config()
    ds_list = _split(datasets, list(cfg.datasets))
    model_list = _split(models, list(cfg.models))
    modes: list[Mode] = list(ALL_MODES) if mode == "all" else [mode]  # type: ignore[list-item]
    rows = build_estimates(cfg, ds_list, model_list, limit or cfg.run.limit, modes)

    t = Table(title="jev-bench estimate (uncached calls only are billed)")
    for col in ("mode", "dataset", "model", "calls", "cached", "in tok/call", "out tok/call",
                "tokens from", "est. cost"):  # fmt: skip
        t.add_column(col, justify="right" if col not in ("mode", "dataset", "model") else "left")
    for r in rows:
        t.add_row(
            r.mode, r.dataset, r.model, str(r.calls), str(r.cached_calls),
            f"{r.input_tokens_per_call:,.0f}", f"{r.output_tokens_per_call:,.0f}", r.source,
            f"${r.cost_usd:.4f}",
        )  # fmt: skip
    console.print(t)

    paid = sum(r.cost_usd for r in rows if r.paid)
    free = sum(r.cost_usd for r in rows if not r.paid)
    ledger = _ledger(cfg)
    console.print(f"Anthropic estimate: [bold]${paid:.4f}[/]   Jev estimate: ${free:.4f}")
    console.print(
        f"Ledger: spent ${ledger.spent:.4f} of ${ledger.cap_usd:.2f} cap, "
        f"remaining ${ledger.remaining:.4f} -> "
        + ("[green]fits[/]" if paid <= ledger.remaining else "[red]EXCEEDS remaining budget[/]")
    )


@app.command()
def budget() -> None:
    """Show recorded Anthropic spend and what's left under the cap."""
    cfg = load_config()
    ledger = _ledger(cfg)
    t = Table(title="Spend ledger")
    t.add_column("model")
    t.add_column("spent", justify="right")
    for model, spent in sorted(ledger.spent_by_model().items()):
        t.add_row(model, f"${spent:.4f}")
    console.print(t)
    console.print(
        f"Total ${ledger.spent:.4f} / cap ${ledger.cap_usd:.2f}  "
        f"remaining [bold]${ledger.remaining:.4f}[/]  ({len(ledger.entries)} entries)"
    )


@app.command()
def run() -> None:
    """Run the benchmark (implemented in phase 3)."""
    raise typer.Exit("`run` is not implemented yet (phase 3).")


@app.command()
def report() -> None:
    """Build summary tables and charts (implemented in phase 4)."""
    raise typer.Exit("`report` is not implemented yet (phase 4).")
