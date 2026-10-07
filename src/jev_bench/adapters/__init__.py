"""Model adapters. Each implements the `Decider` protocol from `jev_bench.types`."""

from __future__ import annotations

from jev_bench.types import Decision, ErrorKind


def check_label(decision: Decision, label_space: tuple[str, ...]) -> Decision:
    """Mark a decision whose label falls outside the label set. It is never dropped."""
    if decision.error is None and decision.label not in label_space:
        return decision.model_copy(
            update={"error": "out_of_label_set", "error_detail": f"label={decision.label!r}"}
        )
    return decision


def failure(kind: ErrorKind, detail: str, **fields: object) -> dict[str, object]:
    return {
        "label": None,
        "confidence": None,
        "error": kind,
        "error_detail": detail[:500],
        **fields,
    }
