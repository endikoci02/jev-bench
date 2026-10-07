"""Accuracy, calibration, selective prediction, latency and cost metrics.

Conventions (also stated in the README):
- Failures (API error, refusal, invalid output, out-of-label-set) count as WRONG
  in accuracy and macro-F1, and are reported separately as failure rates.
- Calibration metrics (ECE, MCE, Brier, AUROC, confident-but-wrong, selective
  curve, hard share) use answered predictions only: a failure has no confidence.
- Brier is top-label Brier, mean (confidence - correct)^2, because Claude only
  gives a confidence for its chosen label. It is comparable across models.
- ECE/MCE use equal-width bins over [0, 1]; bin b holds (b/B, (b+1)/B], with 0
  in the first bin, so a hard 1.0 always lands in the top bin.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

HARD_THRESHOLD = 0.99


def accuracy(correct: BoolArray) -> float:
    return float(np.mean(correct)) if correct.size else math.nan


def macro_f1(gold: Sequence[str], pred: Sequence[str | None], labels: Sequence[str]) -> float:
    """Unweighted mean F1 over classes present in gold or predictions.

    A failed prediction (None) is a false negative for the gold class and a
    false positive for no class.
    """
    present = [c for c in labels if c in set(gold) or c in set(pred)]
    f1s = []
    for c in present:
        tp = sum(g == c and p == c for g, p in zip(gold, pred, strict=True))
        fp = sum(g != c and p == c for g, p in zip(gold, pred, strict=True))
        fn = sum(g == c and p != c for g, p in zip(gold, pred, strict=True))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s)) if f1s else math.nan


def bin_index(conf: FloatArray, n_bins: int) -> NDArray[np.int_]:
    """Bin b covers (b/B, (b+1)/B]; confidence 0 goes to bin 0."""
    return np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)


@dataclass(frozen=True)
class ReliabilityBins:
    lower: FloatArray
    upper: FloatArray
    count: NDArray[np.int_]
    accuracy: FloatArray  # NaN for empty bins
    confidence: FloatArray  # mean confidence, NaN for empty bins


def reliability_bins(conf: FloatArray, correct: BoolArray, n_bins: int = 15) -> ReliabilityBins:
    idx = bin_index(conf, n_bins)
    count = np.bincount(idx, minlength=n_bins)
    acc_sum = np.bincount(idx, weights=correct.astype(float), minlength=n_bins)
    conf_sum = np.bincount(idx, weights=conf, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.where(count > 0, acc_sum / count, np.nan)
        mean_conf = np.where(count > 0, conf_sum / count, np.nan)
    edges = np.linspace(0, 1, n_bins + 1)
    return ReliabilityBins(edges[:-1], edges[1:], count, acc, mean_conf)


def ece(conf: FloatArray, correct: BoolArray, n_bins: int = 15) -> float:
    """Expected calibration error: count-weighted mean |accuracy - confidence| per bin."""
    if conf.size == 0:
        return math.nan
    b = reliability_bins(conf, correct, n_bins)
    nonempty = b.count > 0
    gaps = np.abs(b.accuracy[nonempty] - b.confidence[nonempty])
    return float(np.sum(gaps * b.count[nonempty]) / conf.size)


def mce(conf: FloatArray, correct: BoolArray, n_bins: int = 15) -> float:
    """Maximum calibration error over non-empty bins."""
    if conf.size == 0:
        return math.nan
    b = reliability_bins(conf, correct, n_bins)
    nonempty = b.count > 0
    return float(np.max(np.abs(b.accuracy[nonempty] - b.confidence[nonempty])))


def brier(conf: FloatArray, correct: BoolArray) -> float:
    """Top-label Brier score."""
    return float(np.mean((conf - correct.astype(float)) ** 2)) if conf.size else math.nan


def auroc(conf: FloatArray, correct: BoolArray) -> float:
    """P(conf of a random correct answer > conf of a random wrong one), ties count 1/2.

    NaN when all answers are correct or all wrong. With many hard 1.0
    predictions, ties are common and pull AUROC toward 0.5.
    """
    pos, neg = conf[correct], conf[~correct]
    if pos.size == 0 or neg.size == 0:
        return math.nan
    # Mann-Whitney U via average ranks (handles ties)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty_like(allv)
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2
    return float(u / (pos.size * neg.size))


def confident_wrong_rate(conf: FloatArray, correct: BoolArray, threshold: float) -> float:
    """Share of ALL answered predictions that are wrong with confidence >= threshold."""
    return float(np.mean((conf >= threshold) & ~correct)) if conf.size else math.nan


def hard_share(conf: FloatArray, threshold: float = HARD_THRESHOLD) -> float:
    return float(np.mean(conf >= threshold)) if conf.size else math.nan


@dataclass(frozen=True)
class SelectiveCurve:
    threshold: FloatArray
    coverage: FloatArray
    accuracy: FloatArray


def selective_curve(conf: FloatArray, correct: BoolArray) -> SelectiveCurve:
    """Accuracy on the predictions kept when answering only at conf >= t, for each t.

    Points are taken at unique confidence values, so tied predictions are kept
    or dropped together: a model with mostly hard 1.0s cannot reach low coverage.
    """
    if conf.size == 0:
        empty = np.array([], dtype=float)
        return SelectiveCurve(empty, empty, empty)
    thresholds = np.unique(conf)[::-1]
    cov, acc = [], []
    for t in thresholds:
        keep = conf >= t
        cov.append(keep.mean())
        acc.append(correct[keep].mean())
    return SelectiveCurve(thresholds, np.array(cov), np.array(acc))


def latency_percentiles(latencies_ms: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(latencies_ms, dtype=float)
    if arr.size == 0:
        return {"p50": math.nan, "p95": math.nan, "p99": math.nan}
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99)}


def bootstrap_ci(
    stat: Callable[[NDArray[np.int_]], float],
    n: int,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI. `stat` receives an array of resampled row indices."""
    if n == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    vals = np.array([stat(rng.integers(0, n, n)) for _ in range(n_resamples)])
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return (math.nan, math.nan)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))
