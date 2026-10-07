"""Run orchestration.

- Jev, and every latency run, use concurrent sync calls (bounded per provider).
- Claude accuracy and self-consistency runs use the Message Batches API, submitted
  in chunks. Each chunk is checked against the remaining budget before it is sent,
  and batch IDs are saved to results/state/ so an interrupted run resumes by
  polling the same batches instead of paying for them twice.
- Retries: both SDKs retry 408/429/5xx/connection errors with exponential backoff
  (honouring retry-after). Batch polling adds a tenacity retry on top, since a
  poll loop can run for a long time.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from jev_bench.adapters.claude import ClaudeDecider, aggregate_self_consistency
from jev_bench.adapters.jev import JevDecider
from jev_bench.budget import BudgetCapReachedError, SpendLedger
from jev_bench.cache import atomic_write_json
from jev_bench.config import REPO_ROOT
from jev_bench.estimate import Mode, raw_log_path
from jev_bench.types import Decision, Example

STATE_DIR = REPO_ROOT / "results" / "state"
Progress = Callable[[int], None]


# ---- spend tracking ---------------------------------------------------------


@dataclass
class SpendTracker:
    """Buffers new (uncached) paid spend and writes it to the ledger.

    Flushes every `flush_every_usd`, at the end of a unit, and as soon as
    recorded + pending spend reaches the cap - which then aborts the run.
    """

    ledger: SpendLedger
    run_id: str
    flush_every_usd: float = 0.05
    pending: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    pending_calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_new: float = 0.0

    def add(self, model: str, cost_usd: float, calls: int = 1) -> None:
        self.pending[model] += cost_usd
        self.pending_calls[model] += calls
        self.total_new += cost_usd
        pending_total = sum(self.pending.values())
        if self.ledger.spent + pending_total >= self.ledger.cap_usd:
            self.flush()  # raises BudgetCapReachedError
        elif pending_total >= self.flush_every_usd:
            self.flush()

    def flush(self, note: str = "") -> None:
        models = [m for m, n in self.pending_calls.items() if n]
        for m in models:
            self.ledger.record(
                run_id=self.run_id,
                model=m,
                cost_usd=self.pending[m],
                calls=self.pending_calls[m],
                note=note,
                enforce=False,  # record everything first, then enforce once
            )
            self.pending[m] = 0.0
            self.pending_calls[m] = 0
        if models:
            self.ledger.assert_under_cap()


# ---- raw logs ---------------------------------------------------------------


def write_raw_log(
    *,
    run_id: str,
    dataset: str,
    model: str,
    mode: Mode,
    examples: Sequence[Example],
    decisions: Sequence[Decision],
    warmup: int = 0,
    path: Path | None = None,
) -> Path:
    """One JSONL line per example. Rewritten on every run so it mirrors the latest sample."""
    path = path or raw_log_path(dataset, model, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for i, (ex, d) in enumerate(zip(examples, decisions, strict=True)):
            rec = {
                "run_id": run_id,
                "dataset": dataset,
                "model": model,
                "mode": mode,
                "example_id": ex.id,
                "gold": ex.label,
                "warmup": i < warmup,
                "decision": d.model_dump(mode="json"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


# ---- sync path --------------------------------------------------------------


async def run_sync(
    decider: JevDecider | ClaudeDecider,
    examples: Sequence[Example],
    *,
    concurrency: int,
    tracker: SpendTracker | None,
    namespace: str = "",
    sequential_first: int = 0,
    progress: Progress | None = None,
) -> list[Decision]:
    """Concurrent sync calls, results in input order.

    The first `sequential_first` examples (latency warmup) run one at a time
    before the concurrent phase starts.
    """
    results: list[Decision | None] = [None] * len(examples)
    sem = asyncio.Semaphore(concurrency)

    async def one(i: int) -> None:
        async with sem:
            d = await decider.decide(examples[i], examples[i].label_space, namespace=namespace)
        results[i] = d
        if tracker is not None and not d.cached and d.cost_usd > 0:
            tracker.add(decider.name, d.cost_usd)
        if progress:
            progress(1)

    for i in range(min(sequential_first, len(examples))):
        await one(i)
    try:
        async with asyncio.TaskGroup() as tg:
            for i in range(sequential_first, len(examples)):
                tg.create_task(one(i))
    except* BudgetCapReachedError as eg:
        raise eg.exceptions[0] from None
    return [r for r in results if r is not None]


# ---- batch path -------------------------------------------------------------


@dataclass(frozen=True)
class BatchItem:
    example: Example
    sample_index: int = 0
    temperature: float | None = None


class BatchState:
    """Persisted record of submitted chunks, for resume after interruption."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.chunks: list[dict[str, Any]] = []
        if path.exists():
            self.chunks = json.loads(path.read_text(encoding="utf-8"))["chunks"]

    def save(self) -> None:
        atomic_write_json(self.path, {"chunks": self.chunks})

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


_transient = retry_if_exception_type(
    (anthropic.APIConnectionError, anthropic.InternalServerError, anthropic.RateLimitError)
)


@retry(retry=_transient, wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(8))
def _retrieve(client: anthropic.Anthropic, batch_id: str) -> Any:
    return client.messages.batches.retrieve(batch_id)


@retry(retry=_transient, wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(8))
def _results(client: anthropic.Anthropic, batch_id: str) -> list[dict[str, Any]]:
    return [r.to_dict() for r in client.messages.batches.results(batch_id)]


@retry(retry=_transient, wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(5))
def _submit(client: anthropic.Anthropic, requests: list[dict[str, Any]]) -> Any:
    return client.messages.batches.create(requests=requests)  # type: ignore[arg-type]


def run_batch(
    decider: ClaudeDecider,
    items: Sequence[BatchItem],
    *,
    client: anthropic.Anthropic,
    tracker: SpendTracker,
    state_path: Path,
    cost_per_call: float,
    chunk_size: int = 100,
    poll_seconds: float = 15.0,
    log: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Decision]:
    """Run items through the Batches API. Returns decisions in item order.

    Cached items are not resubmitted. Successful results are cached; per-request
    errors are returned (scored as failures) but not cached, so a rerun retries them.
    """
    keyed = [
        (item, *decider.batch_request(item.example, temperature=item.temperature,
                                      sample_index=item.sample_index))
        for item in items
    ]  # fmt: skip
    by_id = {cid: item for item, cid, _ in keyed}
    fresh: dict[str, Decision] = {}
    state = BatchState(state_path)

    def collect(chunk: dict[str, Any]) -> None:
        batch_id = chunk["batch_id"]
        while True:
            b = _retrieve(client, batch_id)
            counts = b.request_counts
            log(
                f"  batch {batch_id}: {b.processing_status} "
                f"(ok {counts.succeeded}, err {counts.errored}, processing {counts.processing})"
            )
            if b.processing_status == "ended":
                break
            sleep(poll_seconds)
        new_cost = 0.0
        for res in _results(client, batch_id):
            cid = res["custom_id"]
            item = by_id.get(cid)
            if item is None:
                continue
            d = decider.parse_batch_result(res, item.example.label_space, item.sample_index)
            if res["result"]["type"] == "succeeded":
                decider.cache.put(cid, d.model_dump(mode="json"))
            fresh[cid] = d
            new_cost += d.cost_usd
        # Record spend before marking collected: a crash in between re-collects and
        # over-counts (safe for the cap) rather than losing spend.
        try:
            tracker.add(decider.name, new_cost, calls=len(chunk["custom_ids"]))
        finally:
            tracker.flush(note=f"batch {batch_id}")
            chunk["collected"] = True
            state.save()

    # 1. resume anything submitted by an interrupted run
    for chunk in state.chunks:
        if not chunk.get("collected"):
            log(f"  resuming batch {chunk['batch_id']}")
            collect(chunk)

    # 2. submit everything without a usable result. Requests that errored in an
    #    earlier batch are resubmitted (errors are never cached).
    todo = [
        (cid, params)
        for _, cid, params in keyed
        if cid not in decider.cache and not (cid in fresh and fresh[cid].error is None)
    ]
    todo = list(dict(todo).items())  # dedupe identical requests
    for start in range(0, len(todo), chunk_size):
        chunk_reqs = todo[start : start + chunk_size]
        tracker.ledger.check_estimate(cost_per_call * len(chunk_reqs))
        batch = _submit(client, [{"custom_id": cid, "params": p} for cid, p in chunk_reqs])
        chunk = {"batch_id": batch.id, "custom_ids": [cid for cid, _ in chunk_reqs]}
        state.chunks.append(chunk)
        state.save()
        log(f"  submitted batch {batch.id} ({len(chunk_reqs)} requests)")
        collect(chunk)

    state.clear()

    out: list[Decision] = []
    for _item, cid, _ in keyed:
        if cid in fresh:
            out.append(fresh[cid])
        else:
            hit = decider.cache.get(cid)
            if hit is None:  # submitted earlier but its result is gone; should not happen
                raise RuntimeError(f"no result for {cid}; rerun to resubmit")
            out.append(Decision.model_validate({**hit, "cached": True}))
    return out


def aggregate_samples(decisions: Sequence[Decision], k: int) -> list[Decision]:
    """Group k consecutive samples per example and majority-vote them."""
    return [aggregate_self_consistency(decisions[i : i + k]) for i in range(0, len(decisions), k)]
