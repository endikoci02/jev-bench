"""Core data types shared by datasets, adapters, runner and metrics."""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

ConfidenceType = Literal[
    "native_max_prob",  # Jev: max of the returned probability distribution
    "verbalized",  # Claude method A: model states a 0-1 confidence
    "self_consistency",  # Claude method B: agreement rate over k samples
]

ErrorKind = Literal["api_error", "invalid_output", "refusal", "out_of_label_set"]


class Example(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    label: str
    label_space: tuple[str, ...]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class Decision(BaseModel):
    """One model's answer for one example. Failures are kept, never dropped."""

    label: str | None
    confidence: float | None
    confidence_type: ConfidenceType
    probabilities: dict[str, float] | None = None
    api_confidence: float | None = None  # Jev's own concentration statistic
    latency_ms: float | None = None
    cost_usd: float = 0.0
    usage: Usage = Usage()
    model_version: str | None = None
    raw: dict[str, Any] | None = None
    error: ErrorKind | None = None
    error_detail: str | None = None
    cached: bool = False  # served from disk cache: no new spend
    sample_index: int = 0

    @property
    def failed(self) -> bool:
        return self.error is not None


class Decider(Protocol):
    name: str

    async def decide(self, example: Example, label_space: tuple[str, ...]) -> Decision: ...
