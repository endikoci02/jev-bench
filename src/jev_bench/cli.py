"""Command-line interface: `jev-bench estimate | run | report | budget`."""

from __future__ import annotations

import asyncio
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import anthropic
import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from jev_bench.adapters.claude import ClaudeDecider
from jev_bench.budget import BudgetCapReachedError, BudgetExceededError, SpendLedger
from jev_bench.cache import DiskCache
from jev_bench.config import BenchConfig, load_config, load_env
from jev_bench.datasets import load_examples, load_label_spec
from jev_bench.estimate import EstimateRow, Mode, estimate_row
from jev_bench.registry import make_decider
from jev_bench.runner import (
    STATE_DIR,
    BatchItem,
    SpendTracker,
    aggregate_samples,
    run_batch,
    run_sync,
    write_raw_log,
)
from jev_bench.types import Decision, Example

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

ALL_MODES: tuple[Mode, ...] = ("batch", "latency", "self_consistency")


def _split(value: str | None, default: list[str]) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()] if value else default


def _ledger(cfg: BenchConfig) -> SpendLedger:
    return SpendLedger(cfg.resolve(cfg.budget.ledger_path), cfg.budget.cap_usd)


def _cached_count(
    cfg: BenchConfig, model: str, dataset: str, examples: list[Example], mode: Mode
) -> int:
    """How many of the planned calls are already in the disk cache (and so free)."""
    spec = load_label_spec(dataset)
    cache = DiskCache(cfg.resolve(cfg.cache_dir))
    decider = make_decider(cfg, model, spec, cache)
    sc = cfg.run.self_consistency
    k = sc.k if mode == "self_consistency" else 1
    namespace = "latency" if mode == "latency" else ""
    n = 0
    for ex in examples:
        for i in range(k):
            if isinstance(decider, ClaudeDecider):
                temp = sc.temperature if mode == "self_consistency" else None
                params = decider.params(ex, temperature=temp)
                key = decider.key(ex, params, i, namespace)
            else:
                key = decider.key(ex, namespace)
            n += key in cache
    return n


def build_estimates(
    cfg: BenchConfig, datasets: list[str], models: list[str], limit: int, modes: list[Mode]
) -> list[EstimateRow]:
    plans: list[tuple[Mode, str, list[str], int, int]] = []  # mode, dataset, models, n, k
    if "batch" in modes:
        plans += [("batch", ds, models, limit, 1) for ds in datasets]
    if "latency" in modes:
        lat = cfg.run.latency
        plans.append(("latency", lat.dataset, models, lat.calls_per_model + lat.warmup_calls, 1))
    if "self_consistency" in modes:
        sc = cfg.run.self_consistency
        if sc.model in models:
            plans.append(("self_consistency", sc.dataset, [sc.model], sc.n_examples, sc.k))

    rows: list[EstimateRow] = []
    for mode, ds, model_list, n, k in plans:
        spec = load_label_spec(ds)
        examples = load_examples(ds, cfg, n)
        for m in model_list:
            rows.append(
                estimate_row(
                    cfg, dataset=ds, model=m, mode=mode, spec=spec, examples=examples,
                    calls=len(examples) * k,
                    cached_calls=_cached_count(cfg, m, ds, examples, mode),
                )
            )  # fmt: skip
    return rows


def print_estimate(cfg: BenchConfig, rows: list[EstimateRow]) -> float:
    """Print the estimate table; return the Anthropic (paid) total."""
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
    return paid


@app.command()
def estimate(
    datasets: Annotated[str | None, typer.Option(help="Comma-separated; default all")] = None,
    models: Annotated[str | None, typer.Option(help="Comma-separated; default all")] = None,
    limit: Annotated[int | None, typer.Option(help="Examples per dataset")] = None,
    dev: Annotated[bool, typer.Option("--dev", help="Use run.dev_limit examples")] = False,
    mode: Annotated[str, typer.Option(help="batch, latency, self_consistency or all")] = "all",
) -> None:
    """Show call counts and estimated cost per model before anything runs."""
    cfg = load_config()
    ds_list = _split(datasets, list(cfg.datasets))
    model_list = _split(models, list(cfg.models))
    modes: list[Mode] = list(ALL_MODES) if mode == "all" else [mode]  # type: ignore[list-item]
    n = cfg.run.dev_limit if dev else (limit or cfg.run.limit)
    print_estimate(cfg, build_estimates(cfg, ds_list, model_list, n, modes))


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
def run(
    datasets: Annotated[str | None, typer.Option(help="Comma-separated; default all")] = None,
    models: Annotated[str | None, typer.Option(help="Comma-separated; default all")] = None,
    limit: Annotated[int | None, typer.Option(help="Examples per dataset")] = None,
    dev: Annotated[bool, typer.Option("--dev", help="Use run.dev_limit examples")] = False,
    latency: Annotated[bool, typer.Option("--latency", help="Sync latency run")] = False,
    self_consistency: Annotated[
        bool, typer.Option("--self-consistency", help="Haiku k-sample run")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the spend confirmation")] = False,
) -> None:
    """Run the benchmark. Default mode: accuracy (Jev sync, Claude via the Batches API)."""
    if latency and self_consistency:
        raise typer.BadParameter("choose at most one of --latency / --self-consistency")
    mode: Mode = "latency" if latency else "self_consistency" if self_consistency else "batch"
    cfg = load_config()
    load_env()
    model_list = _split(models, list(cfg.models))
    if mode == "latency":
        ds_list = [cfg.run.latency.dataset]
    elif mode == "self_consistency":
        ds_list = [cfg.run.self_consistency.dataset]
        model_list = [cfg.run.self_consistency.model]
    else:
        ds_list = _split(datasets, list(cfg.datasets))
    n = cfg.run.dev_limit if dev else (limit or cfg.run.limit)
    _require_keys(cfg, model_list)

    rows = build_estimates(cfg, ds_list, model_list, n, [mode])
    paid = print_estimate(cfg, rows)
    ledger = _ledger(cfg)
    try:
        ledger.check_estimate(paid)
    except BudgetExceededError as e:
        console.print(f"[red]Refusing to start:[/] {e}")
        raise typer.Exit(1) from None
    if paid > 0 and not yes and not typer.confirm(f"Spend up to ~${paid:.4f}?"):
        raise typer.Exit(1)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    tracker = SpendTracker(ledger, run_id)
    per_call = {(r.dataset, r.model): r.cost_usd / max(1, r.calls - r.cached_calls) for r in rows}
    console.print(f"run_id {run_id}  mode={mode}  n={n}")
    try:
        for ds in ds_list:
            _run_dataset(cfg, ds, model_list, n, mode, run_id, tracker, per_call)
    except (BudgetCapReachedError, BudgetExceededError) as e:
        console.print(f"[red]ABORTED:[/] {e}")
        raise typer.Exit(2) from None
    finally:
        tracker.flush(note="end of run")
    console.print(
        f"New Anthropic spend this run: ${tracker.total_new:.4f}. "
        f"Ledger total ${ledger.spent:.4f}, remaining ${ledger.remaining:.4f}."
    )


def _require_keys(cfg: BenchConfig, models: list[str]) -> None:
    env_var = {"typesafe": "TYPESAFE_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
    missing = sorted(
        {env_var[cfg.models[m].provider] for m in models if not os.environ.get(
            env_var[cfg.models[m].provider])}
    )  # fmt: skip
    if missing:
        raise typer.BadParameter(f"missing in .env: {', '.join(missing)}")


def _run_dataset(
    cfg: BenchConfig,
    ds: str,
    models: list[str],
    n: int,
    mode: Mode,
    run_id: str,
    tracker: SpendTracker,
    per_call: dict[tuple[str, str], float],
) -> None:
    spec = load_label_spec(ds)
    cache = DiskCache(cfg.resolve(cfg.cache_dir))
    warmup = cfg.run.latency.warmup_calls if mode == "latency" else 0
    if mode == "latency":
        n = cfg.run.latency.calls_per_model + warmup
    elif mode == "self_consistency":
        n = cfg.run.self_consistency.n_examples
    examples = load_examples(ds, cfg, n)

    for model in models:
        m = cfg.models[model]
        decider = make_decider(cfg, model, spec, cache)
        console.print(f"[bold]{ds} / {model}[/] ({len(examples)} examples)")
        decisions: list[Decision]
        if isinstance(decider, ClaudeDecider) and mode in ("batch", "self_consistency"):
            sc = cfg.run.self_consistency
            items = (
                [BatchItem(ex, i, sc.temperature) for ex in examples for i in range(sc.k)]
                if mode == "self_consistency"
                else [BatchItem(ex) for ex in examples]
            )
            samples = run_batch(
                decider,
                items,
                client=anthropic.Anthropic(max_retries=5),
                tracker=tracker,
                state_path=STATE_DIR / f"{ds}__{model}__{mode}.json",
                cost_per_call=per_call.get((ds, model), 0.0),
                log=console.print,
            )
            decisions = aggregate_samples(samples, sc.k) if mode == "self_consistency" else samples
        else:
            with Progress(console=console, transient=True) as bar:
                task = bar.add_task(model, total=len(examples))
                decisions = asyncio.run(
                    run_sync(
                        decider,
                        examples,
                        concurrency=cfg.run.concurrency[m.provider],
                        tracker=tracker if m.provider == "anthropic" else None,
                        namespace="latency" if mode == "latency" else "",
                        sequential_first=warmup,
                        progress=lambda k, t=task: bar.advance(t, k),
                    )
                )
        path = write_raw_log(
            run_id=run_id, dataset=ds, model=model, mode=mode,
            examples=examples, decisions=decisions, warmup=warmup,
        )  # fmt: skip
        _print_unit_summary(examples, decisions, path, warmup)


def _print_unit_summary(
    examples: list[Example], decisions: list[Decision], path: Path, warmup: int
) -> None:
    scored = list(zip(examples, decisions, strict=True))[warmup:]
    correct = sum(d.error is None and d.label == ex.label for ex, d in scored)
    errors = Counter(d.error for _, d in scored if d.error)
    cached = sum(d.cached for _, d in scored)
    new_cost = sum(d.cost_usd for _, d in scored if not d.cached)
    console.print(
        f"  accuracy {correct}/{len(scored)} = {correct / max(1, len(scored)):.1%}  "
        f"failures {dict(errors) or 0}  cached {cached}  new cost ${new_cost:.5f}  -> {path.name}"
    )


@app.command()
def report() -> None:
    """Build summary tables and charts (implemented in phase 4)."""
    raise typer.Exit("`report` is not implemented yet (phase 4).")
