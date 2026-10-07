"""Turn raw per-call logs into summary.csv, summary.md and charts."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jev_bench import metrics as M
from jev_bench.config import REPO_ROOT, BenchConfig
from jev_bench.datasets import load_label_spec
from jev_bench.estimate import RAW_DIR
from jev_bench.types import Decision

RESULTS_DIR = REPO_ROOT / "results"
CHART_DIR = RESULTS_DIR / "charts"

# Fixed categorical order (dataviz reference palette, slots 1-3), plus a marker
# per model so identity never depends on color alone.
STYLE = {
    "jev": {"color": "#2a78d6", "marker": "o", "name": "Jev"},
    "haiku": {"color": "#eb6834", "marker": "s", "name": "Haiku 4.5"},
    "sonnet": {"color": "#1baf7a", "marker": "^", "name": "Sonnet 5.5"},
}
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


@dataclass
class Unit:
    """All logged decisions for one (dataset, model, mode, confidence source)."""

    dataset: str
    model: str
    mode: str
    method: str  # human label, e.g. "verbalized"
    gold: list[str]
    decisions: list[Decision]
    warmup: list[bool]
    example_ids: list[str]

    @property
    def key(self) -> str:
        return f"{self.model}|{self.method}"

    @property
    def label(self) -> str:
        return f"{STYLE.get(self.model, {'name': self.model})['name']} ({self.method})"


METHOD = {
    ("jev", "batch"): "native max-prob",
    ("haiku", "batch"): "verbalized",
    ("sonnet", "batch"): "verbalized",
    ("haiku", "self_consistency"): "self-consistency",
}


def _model_order(path: Path) -> tuple[int, str]:
    """Jev first, then Claude models in the fixed STYLE order."""
    model = path.stem.split("__")[1]
    return (list(STYLE).index(model) if model in STYLE else len(STYLE), path.stem)


def load_units(raw_dir: Path = RAW_DIR) -> list[Unit]:
    units: list[Unit] = []
    for path in sorted(raw_dir.glob("*.jsonl"), key=_model_order):
        dataset, model, mode = path.stem.split("__")
        gold, decisions, warm, ids = [], [], [], []
        with path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                gold.append(r["gold"])
                decisions.append(Decision.model_validate(r["decision"]))
                warm.append(bool(r.get("warmup")))
                ids.append(r["example_id"])
        method = METHOD.get((model, mode), mode)
        units.append(Unit(dataset, model, mode, method, gold, decisions, warm, ids))
        if model == "jev" and mode == "batch":
            # Same predictions, scored with the API's own `confidence` field.
            api = [
                d.model_copy(update={"confidence": d.api_confidence}) if d.error is None else d
                for d in decisions
            ]
            units.append(Unit(dataset, model, mode, "API confidence", gold, api, warm, ids))
    return units


def _arrays(unit: Unit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(all-correct mask, answered confidences, answered-correct mask), warmup excluded."""
    rows = [(g, d) for g, d, w in zip(unit.gold, unit.decisions, unit.warmup, strict=True) if not w]
    correct_all = np.array([d.error is None and d.label == g for g, d in rows], dtype=bool)
    answered = [(g, d) for g, d in rows if d.error is None and d.confidence is not None]
    conf = np.array([d.confidence for _, d in answered], dtype=float)
    corr = np.array([d.label == g for g, d in answered], dtype=bool)
    return correct_all, conf, corr


def summarize(unit: Unit, cfg: BenchConfig) -> dict[str, Any]:
    rows = [(g, d) for g, d, w in zip(unit.gold, unit.decisions, unit.warmup, strict=True) if not w]
    correct_all, conf, corr = _arrays(unit)
    n = len(rows)
    errors = defaultdict(int)
    for _, d in rows:
        if d.error:
            errors[d.error] += 1
    spec_labels = load_label_spec(unit.dataset).label_space
    boot = cfg.run.bootstrap.n_resamples
    acc_ci = M.bootstrap_ci(lambda i: M.accuracy(correct_all[i]), n, n_resamples=boot, seed=1)
    ece_ci = M.bootstrap_ci(
        lambda i: M.ece(conf[i], corr[i], cfg.run.ece_bins), conf.size, n_resamples=boot, seed=2
    )
    m = cfg.models.get(unit.model)
    usages = [d.usage for _, d in rows]
    k = cfg.run.self_consistency.k if unit.mode == "self_consistency" else 1

    def per_1k(batch: bool) -> float:
        if m is None or not usages or (batch and m.batch_discount == 0):
            return math.nan  # unknown model, no data, or no batch API (Jev)
        return 1000 * sum(m.cost(u, batch=batch) for u in usages) / len(usages)

    lat = M.latency_percentiles([d.latency_ms for _, d in rows if d.latency_ms is not None])
    return {
        "dataset": unit.dataset,
        "model": unit.model,
        "mode": unit.mode,
        "confidence_method": unit.method,
        "model_version": next((d.model_version for _, d in rows if d.model_version), ""),
        "n": n,
        "answered": int(conf.size),
        "accuracy": M.accuracy(correct_all),
        "accuracy_ci_lo": acc_ci[0],
        "accuracy_ci_hi": acc_ci[1],
        "macro_f1": M.macro_f1(
            [g for g, _ in rows],
            [d.label if d.error is None else None for _, d in rows],
            spec_labels,
        ),
        "ece": M.ece(conf, corr, cfg.run.ece_bins),
        "ece_ci_lo": ece_ci[0],
        "ece_ci_hi": ece_ci[1],
        "mce": M.mce(conf, corr, cfg.run.ece_bins),
        "brier": M.brier(conf, corr),
        "auroc": M.auroc(conf, corr),
        "confident_wrong_0.8": M.confident_wrong_rate(conf, corr, 0.8),
        "confident_wrong_0.9": M.confident_wrong_rate(conf, corr, 0.9),
        "hard_share": M.hard_share(conf),
        "failure_rate": sum(errors.values()) / n if n else math.nan,
        "failures": dict(errors),
        "latency_p50_ms": lat["p50"],
        "latency_p95_ms": lat["p95"],
        "latency_p99_ms": lat["p99"],
        "calls_per_decision": k,
        "cost_per_1k_standard": per_1k(False),
        "cost_per_1k_batch": per_1k(True),
    }


def build_summary(units: list[Unit], cfg: BenchConfig) -> pd.DataFrame:
    df = pd.DataFrame([summarize(u, cfg) for u in units])
    if df.empty:
        return df
    # Attach each model's latency-run percentiles to its accuracy rows.
    lat = df[df["mode"] == "latency"].set_index("model")
    for col in ("latency_p50_ms", "latency_p95_ms", "latency_p99_ms"):
        df.loc[df["mode"] != "latency", col] = df.loc[df["mode"] != "latency", "model"].map(
            lat.get(col, {})
        )
    return df


# ---- markdown -----------------------------------------------------------------


def _fmt(v: Any, kind: str = "f3") -> str:
    if isinstance(v, float) and math.isnan(v):
        return "n/a"
    if kind == "pct":
        return f"{v:.1%}"
    if kind == "usd":
        return f"${v:.3f}"
    if kind == "ms":
        return f"{v:,.0f}"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def summary_markdown(df: pd.DataFrame, cfg: BenchConfig) -> str:
    acc = df[df["mode"] != "latency"]
    lines = [
        "# jev-bench summary",
        "",
        "Accuracy and macro-F1 count failures as wrong. Calibration metrics use answered",
        "predictions. Brier is top-label. 95% CIs: percentile bootstrap "
        f"({cfg.run.bootstrap.n_resamples} resamples). Cost per 1k decisions from logged usage.",
        "Claude has no native probabilities here: its confidence is *verbalized* or "
        "*self-consistency*, a weaker signal than Jev's distribution.",
        "",
    ]
    for ds, g in acc.groupby("dataset", sort=False):
        lines += [
            f"## {ds}",
            "",
            "| model (confidence) | n | accuracy [95% CI] | macro-F1 | ECE [95% CI] | MCE | Brier "
            "| AUROC | wrong@≥0.8 | wrong@≥0.9 | hard ≥0.99 | failures | $/1k std | $/1k batch |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _, r in g.iterrows():
            name = (
                f"{STYLE.get(r['model'], {'name': r['model']})['name']} ({r['confidence_method']})"
            )
            lines.append(
                f"| {name} | {r['n']} "
                f"| {_fmt(r['accuracy'], 'pct')} [{_fmt(r['accuracy_ci_lo'], 'pct')}, "
                f"{_fmt(r['accuracy_ci_hi'], 'pct')}] | {_fmt(r['macro_f1'])} "
                f"| {_fmt(r['ece'])} [{_fmt(r['ece_ci_lo'])}, {_fmt(r['ece_ci_hi'])}] "
                f"| {_fmt(r['mce'])} | {_fmt(r['brier'])} | {_fmt(r['auroc'])} "
                f"| {_fmt(r['confident_wrong_0.8'], 'pct')} "
                f"| {_fmt(r['confident_wrong_0.9'], 'pct')} "
                f"| {_fmt(r['hard_share'], 'pct')} | {_fmt(r['failure_rate'], 'pct')} "
                f"| {_fmt(r['cost_per_1k_standard'], 'usd')} "
                f"| {_fmt(r['cost_per_1k_batch'], 'usd')} |"
            )
        lines.append("")
    lat = df[df["mode"] == "latency"]
    if not lat.empty:
        lines += [
            "## Latency (sync Messages API / Jev API)",
            "",
            f"Client: {cfg.run.latency.client_location}. Concurrency per provider: "
            f"{dict(cfg.run.concurrency)}. {cfg.run.latency.warmup_calls} warmup calls excluded. "
            f"Dataset: {cfg.run.latency.dataset}.",
            "",
            "| model | n | p50 ms | p95 ms | p99 ms | failures |",
            "|---|---|---|---|---|---|",
        ]
        for _, r in lat.iterrows():
            lines.append(
                f"| {STYLE.get(r['model'], {'name': r['model']})['name']} | {r['n']} "
                f"| {_fmt(r['latency_p50_ms'], 'ms')} | {_fmt(r['latency_p95_ms'], 'ms')} "
                f"| {_fmt(r['latency_p99_ms'], 'ms')} | {_fmt(r['failure_rate'], 'pct')} |"
            )
    return "\n".join(lines) + "\n"


def same_subset_markdown(units: list[Unit], n_bins: int) -> str:
    """Compare every method on exactly the examples the self-consistency run used.

    Self-consistency runs on a smaller subset (it costs k calls per example), so
    its row in the main table is not directly comparable to 200-example rows.
    """
    sc = [u for u in units if u.mode == "self_consistency"]
    if not sc:
        return ""
    lines = []
    for s in sc:
        ids = {d_id for d_id, w in zip(s.example_ids, s.warmup, strict=True) if not w}
        lines += [
            f"## Same-subset comparison: {s.dataset} "
            f"({len(ids)} examples used by self-consistency)",
            "",
            "| model (confidence) | accuracy | ECE | AUROC | wrong@≥0.9 | hard ≥0.99 |",
            "|---|---|---|---|---|---|",
        ]
        for u in units:
            if u.dataset != s.dataset or u.mode == "latency":
                continue
            keep = [i for i, d_id in enumerate(u.example_ids) if d_id in ids]
            sub = Unit(
                u.dataset, u.model, u.mode, u.method,
                [u.gold[i] for i in keep], [u.decisions[i] for i in keep],
                [u.warmup[i] for i in keep], [u.example_ids[i] for i in keep],
            )  # fmt: skip
            correct_all, conf, corr = _arrays(sub)
            lines.append(
                f"| {u.label} | {_fmt(M.accuracy(correct_all), 'pct')} "
                f"| {_fmt(M.ece(conf, corr, n_bins))} | {_fmt(M.auroc(conf, corr))} "
                f"| {_fmt(M.confident_wrong_rate(conf, corr, 0.9), 'pct')} "
                f"| {_fmt(M.hard_share(conf), 'pct')} |"
            )
        lines.append("")
    return "\n".join(lines)


# ---- charts -------------------------------------------------------------------


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=11)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def reliability_chart(units: list[Unit], dataset: str, n_bins: int, out: Path) -> Path:
    us = [u for u in units if u.dataset == dataset and u.mode != "latency"]
    fig = plt.figure(figsize=(4.2 * len(us), 5.6), facecolor=SURFACE)
    grid = fig.add_gridspec(2, len(us), height_ratios=[3, 1], hspace=0.08, wspace=0.25)
    for j, u in enumerate(us):
        _, conf, corr = _arrays(u)
        b = M.reliability_bins(conf, corr, n_bins)
        color = STYLE[u.model]["color"]
        ax = fig.add_subplot(grid[0, j])
        _style_axes(ax)
        width = 1 / n_bins
        nonempty = b.count > 0
        ax.bar(
            b.lower[nonempty], b.accuracy[nonempty], width=width * 0.9, align="edge",
            color=color, alpha=0.9 if u.method != "API confidence" else 0.55,
        )  # fmt: skip
        ax.plot([0, 1], [0, 1], linestyle="--", color=TEXT_2, linewidth=1.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticklabels([])
        ax.set_title(u.label, color=TEXT, fontsize=13, loc="left")
        if j == 0:
            ax.set_ylabel("accuracy in bin", color=TEXT_2, fontsize=12)
        ax.text(
            0.03, 0.97,
            f"ECE {M.ece(conf, corr, n_bins):.3f}\nhard ≥0.99: {M.hard_share(conf):.0%}\n"
            f"n = {conf.size}",
            transform=ax.transAxes, va="top", fontsize=11, color=TEXT, zorder=5,
            bbox={"facecolor": SURFACE, "edgecolor": GRID, "boxstyle": "round", "alpha": 0.9},
        )  # fmt: skip
        hx = fig.add_subplot(grid[1, j], sharex=ax)
        _style_axes(hx)
        hx.bar(b.lower, b.count, width=width * 0.9, align="edge", color=color, alpha=0.6)
        hx.set_xlabel("confidence", color=TEXT_2, fontsize=12)
        if j == 0:
            hx.set_ylabel("count", color=TEXT_2, fontsize=12)
    fig.suptitle(
        f"Reliability: {dataset} (dashed = perfect calibration)", color=TEXT, fontsize=15,
        x=0.01, ha="left",
    )  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def selective_chart(units: list[Unit], dataset: str, out: Path) -> Path:
    us = [u for u in units if u.dataset == dataset and u.mode != "latency"]
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    _style_axes(ax)
    lowest = 1.0
    for u in us:
        _, conf, corr = _arrays(u)
        c = M.selective_curve(conf, corr)
        if c.coverage.size == 0:
            continue
        s = STYLE[u.model]
        ls = ":" if u.method in ("API confidence", "self-consistency") else "-"
        ax.plot(c.coverage, c.accuracy, ls, color=s["color"], marker=s["marker"],
                markersize=8, linewidth=2, label=u.label)  # fmt: skip
        lowest = min(lowest, float(c.accuracy.min()))
    # Curves all end at coverage 1.0, so end labels would collide: use a legend.
    ax.legend(frameon=False, fontsize=11, labelcolor=TEXT, loc="lower left")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(max(0.0, lowest - 0.05), 1.01)
    ax.set_xlabel("coverage (share of examples answered)", color=TEXT_2, fontsize=12)
    ax.set_ylabel("accuracy on answered", color=TEXT_2, fontsize=12)
    ax.set_title(
        f"Selective accuracy: {dataset}\nAnswer only when confidence ≥ t; ties move together",
        color=TEXT, fontsize=14, loc="left",
    )  # fmt: skip
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def scatter_chart(
    df: pd.DataFrame, x: str, xlabel: str, title: str, out: Path, unit: str = ""
) -> Path | None:
    rows = df[(df["mode"] == "batch") & (df["confidence_method"] != "API confidence")]
    rows = rows[rows[x].notna()]
    if rows.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    _style_axes(ax)
    for model, g in rows.groupby("model", sort=False):
        s = STYLE[model]
        ax.scatter(g[x], g["accuracy"], color=s["color"], marker=s["marker"], s=90,
                   edgecolor=SURFACE, linewidth=2, label=s["name"], zorder=3)  # fmt: skip
        for _, r in g.iterrows():
            ax.annotate(r["dataset"], (r[x], r["accuracy"]), xytext=(7, 4),
                        textcoords="offset points", fontsize=10, color=TEXT_2)  # fmt: skip
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{unit}{v:g}"))
    ax.set_ylim(max(0.0, float(rows["accuracy"].min()) - 0.05), 1.01)
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=12)
    ax.set_ylabel("accuracy", color=TEXT_2, fontsize=12)
    ax.set_title(title, color=TEXT, fontsize=14, loc="left")
    ax.legend(frameon=False, fontsize=11, labelcolor=TEXT)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def write_report(cfg: BenchConfig, raw_dir: Path = RAW_DIR, out_dir: Path = RESULTS_DIR) -> dict:
    units = load_units(raw_dir)
    df = build_summary(units, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "summary.csv", index=False)
    md = summary_markdown(df, cfg)
    subset = same_subset_markdown(units, cfg.run.ece_bins)
    (out_dir / "summary.md").write_text(md + ("\n" + subset if subset else ""), encoding="utf-8")
    charts: list[Path] = []
    chart_dir = out_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    for ds in dict.fromkeys(u.dataset for u in units if u.mode != "latency"):
        charts.append(
            reliability_chart(units, ds, cfg.run.ece_bins, chart_dir / f"reliability_{ds}.png")
        )
        charts.append(selective_chart(units, ds, chart_dir / f"selective_{ds}.png"))
    for path in (
        scatter_chart(df, "latency_p50_ms", "p50 latency, ms (log; sync API from Tirana)",
                      "Accuracy vs latency (latency measured on SST-2 prompts)",
                      chart_dir / "accuracy_vs_latency.png", unit=""),
        scatter_chart(df, "cost_per_1k_standard", "USD per 1k decisions (log; standard prices)",
                      "Accuracy vs cost", chart_dir / "accuracy_vs_cost.png", unit="$"),
    ):  # fmt: skip
        if path:
            charts.append(path)
    return {"rows": len(df), "charts": charts}
