"""Persistent spend ledger with a hard cap on cumulative Anthropic spend."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from jev_bench.cache import atomic_write_json

_EPS = 1e-9  # absorb float rounding in dollar sums


class BudgetExceededError(RuntimeError):
    """A run's estimate does not fit in the remaining budget."""


class BudgetCapReachedError(RuntimeError):
    """Recorded spend has reached the cap; the run must stop."""


class LedgerEntry(BaseModel):
    timestamp: str
    run_id: str
    model: str
    cost_usd: float
    calls: int
    note: str = ""


class LedgerState(BaseModel):
    cap_usd: float
    entries: list[LedgerEntry] = []


class SpendLedger:
    """Tracks real (usage-based) spend. Only paid providers are recorded here.

    The cap from config always wins over any value stored in the file, so
    editing the JSON cannot raise the limit.
    """

    def __init__(self, path: Path, cap_usd: float) -> None:
        self.path = path
        self.cap_usd = cap_usd
        self._lock = threading.Lock()
        if path.exists():
            with path.open(encoding="utf-8") as f:
                self._state = LedgerState.model_validate(json.load(f))
            self._state.cap_usd = cap_usd
        else:
            self._state = LedgerState(cap_usd=cap_usd)

    @property
    def spent(self) -> float:
        return sum(e.cost_usd for e in self._state.entries)

    @property
    def remaining(self) -> float:
        return max(0.0, self.cap_usd - self.spent)

    @property
    def entries(self) -> list[LedgerEntry]:
        return list(self._state.entries)

    def spent_by_model(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for e in self._state.entries:
            out[e.model] = out.get(e.model, 0.0) + e.cost_usd
        return out

    def check_estimate(self, estimate_usd: float) -> None:
        """Refuse to start a run whose estimate exceeds the remaining budget."""
        if estimate_usd > self.remaining + _EPS:
            raise BudgetExceededError(
                f"Estimated ${estimate_usd:.4f} exceeds remaining budget "
                f"${self.remaining:.4f} (cap ${self.cap_usd:.2f}, spent ${self.spent:.4f})."
            )

    def record(
        self, *, run_id: str, model: str, cost_usd: float, calls: int, note: str = ""
    ) -> None:
        """Persist real spend, then raise if the cap is reached."""
        if cost_usd < 0:
            raise ValueError("cost_usd must be non-negative")
        with self._lock:
            self._state.entries.append(
                LedgerEntry(
                    timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
                    run_id=run_id,
                    model=model,
                    cost_usd=cost_usd,
                    calls=calls,
                    note=note,
                )
            )
            atomic_write_json(self.path, self._state.model_dump())
        self.assert_under_cap()

    def assert_under_cap(self) -> None:
        if self.spent >= self.cap_usd - _EPS:
            raise BudgetCapReachedError(
                f"Spend ${self.spent:.4f} reached the cap ${self.cap_usd:.2f}; aborting."
            )
