"""Claude prompt + output schema, built from the same LabelSpec Jev receives."""

from __future__ import annotations

from typing import Any

from jev_bench.datasets import LabelSpec

PROMPT_VERSION = "v1"  # bump when wording changes; part of the cache key


def system_prompt(spec: LabelSpec) -> str:
    options = "\n".join(f"- {label}: {desc}" for label, desc in spec.labels.items())
    return (
        "You are a text classifier.\n"
        f"Task: {spec.instructions}\n\n"
        f"Options:\n{options}\n\n"
        "Reply with JSON only: the single best option as `label` (copied exactly), and "
        "`confidence`, your probability from 0 to 1 that the label is correct. "
        "No explanation."
    )


def output_schema(spec: LabelSpec) -> dict[str, Any]:
    # Numeric bounds are unsupported by structured outputs; 0..1 is checked client-side.
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": list(spec.labels)},
            "confidence": {"type": "number"},
        },
        "required": ["label", "confidence"],
        "additionalProperties": False,
    }
