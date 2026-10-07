"""TypeSafe Jev adapter: one Choice question per example, native probabilities."""

from __future__ import annotations

import time
from typing import Any

import httpx2
from typesafe_sdk import AsyncTypeSafeClient, Choice, RetryPolicy, TypeSafeAPIError, TypeSafeError

from jev_bench.adapters import check_label, failure
from jev_bench.cache import DiskCache, cache_key
from jev_bench.config import ModelConfig
from jev_bench.datasets import LabelSpec
from jev_bench.types import Decision, Example, Usage

QUESTION_ID = "label"


class JevDecider:
    def __init__(
        self,
        name: str,
        model: ModelConfig,
        spec: LabelSpec,
        cache: DiskCache,
        *,
        client: AsyncTypeSafeClient | None = None,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.spec = spec
        self.cache = cache
        self._client = client
        self._transport = transport

    @property
    def client(self) -> AsyncTypeSafeClient:
        # Created lazily: the SDK validates the API key on construction, and
        # key-free code paths (estimate, cache hits, tests) must not need one.
        if self._client is None:
            self._client = AsyncTypeSafeClient(
                model=self.model.model_id,
                retry=RetryPolicy(max_retries=5, backoff_max=30.0),
                timeout=60.0,
                transport=self._transport,
            )
        return self._client

    def payload(self, example: Example) -> dict[str, Any]:
        return {
            "state": example.text,
            "questions": {
                QUESTION_ID: {"instructions": self.spec.instructions, "criteria": self.spec.labels}
            },
        }

    def key(self, example: Example, namespace: str = "") -> str:
        payload = self.payload(example)
        if namespace:
            payload["namespace"] = namespace
        return cache_key(provider="typesafe", model=self.model.model_id, payload=payload, params={})

    async def decide(
        self, example: Example, label_space: tuple[str, ...], *, namespace: str = ""
    ) -> Decision:
        key = self.key(example, namespace)
        hit = self.cache.get(key)
        if hit is not None:
            return Decision.model_validate({**hit, "cached": True})

        start = time.perf_counter()
        try:
            resp = await self.client.system_one(
                state=example.text,
                questions={
                    QUESTION_ID: Choice(
                        instructions=self.spec.instructions, criteria=self.spec.labels
                    )
                },
            )
        except TypeSafeError as e:
            # Raised after the SDK's own retries. Not cached, so a rerun retries it.
            status = f" status={e.status}" if isinstance(e, TypeSafeAPIError) else ""
            return Decision(
                confidence_type="native_max_prob",
                **failure("api_error", f"{type(e).__name__}{status}: {e}"),  # type: ignore[arg-type]
            )
        latency_ms = (time.perf_counter() - start) * 1000

        decision = parse_response(resp.model_dump(mode="json"), self.model, latency_ms)
        decision = check_label(decision, label_space)
        self.cache.put(key, decision.model_dump(mode="json"))
        return decision


def parse_response(raw: dict[str, Any], model: ModelConfig, latency_ms: float) -> Decision:
    usage = Usage(
        input_tokens=raw.get("usage", {}).get("input_tokens") or 0,
        output_tokens=raw.get("usage", {}).get("output_tokens") or 0,
    )
    base: dict[str, Any] = {
        "confidence_type": "native_max_prob",
        "latency_ms": latency_ms,
        "usage": usage,
        "cost_usd": model.cost(usage),
        "model_version": raw.get("model"),
        "raw": raw,
    }
    answer = raw.get("answers", {}).get(QUESTION_ID)
    if not answer or not answer.get("probabilities"):
        return Decision(**base, **failure("invalid_output", "missing choice answer"))  # type: ignore[arg-type]

    probs = {k: float(v) for k, v in answer["probabilities"].items()}
    return Decision(
        **base,
        label=answer.get("choice"),
        confidence=max(probs.values()),
        probabilities=probs,
        api_confidence=answer.get("confidence"),
    )
