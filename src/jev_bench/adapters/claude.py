"""Claude-as-classifier adapter (any Claude model ID), sync and batch modes.

Output is constrained with structured outputs (`output_config.format`) to
`{label: enum, confidence: number}`. Forced tool use is not an option: Sonnet 5.5
rejects `tool_choice` any/tool. Confidence is *verbalized*, not a probability.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Sequence
from typing import Any

import anthropic

from jev_bench.adapters import check_label, failure
from jev_bench.cache import DiskCache, cache_key
from jev_bench.config import ModelConfig
from jev_bench.datasets import LabelSpec
from jev_bench.prompts import PROMPT_VERSION, output_schema, system_prompt
from jev_bench.types import ConfidenceType, Decision, Example, Usage


class ClaudeDecider:
    def __init__(
        self,
        name: str,
        model: ModelConfig,
        spec: LabelSpec,
        cache: DiskCache,
        *,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.spec = spec
        self.cache = cache
        self._client = client
        self._system = system_prompt(spec)
        self._schema = output_schema(spec)

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(max_retries=5, timeout=60.0)
        return self._client

    # ---- request building -------------------------------------------------

    def params(
        self, example: Example, *, temperature: float | None = None, use_cache: bool = False
    ) -> dict[str, Any]:
        """Messages API params. `temperature` overrides the model default (self-consistency)."""
        system: list[dict[str, Any]] = [{"type": "text", "text": self._system}]
        if use_cache and self.model.prompt_caching:
            system[0]["cache_control"] = {"type": "ephemeral"}
        p: dict[str, Any] = {
            "model": self.model.model_id,
            "max_tokens": self.model.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": example.text}],
            "output_config": {"format": {"type": "json_schema", "schema": self._schema}},
        }
        temp = temperature if temperature is not None else self.model.temperature
        if temp is not None:
            p["temperature"] = temp
        if self.model.thinking is not None:
            p["thinking"] = {"type": self.model.thinking}
        return p

    def key(
        self, example: Example, params: dict[str, Any], sample_index: int = 0, namespace: str = ""
    ) -> str:
        # cache_control only changes billing, not the answer, so it is excluded.
        system_text = params["system"][0]["text"]
        payload = {"system": system_text, "user": example.text, "prompt": PROMPT_VERSION}
        if namespace:
            payload["namespace"] = namespace
        return cache_key(
            provider="anthropic",
            model=self.model.model_id,
            payload=payload,
            params={k: v for k, v in params.items() if k not in ("system", "messages", "model")},
            sample_index=sample_index,
        )

    # ---- sync mode --------------------------------------------------------

    async def decide(
        self,
        example: Example,
        label_space: tuple[str, ...],
        *,
        temperature: float | None = None,
        sample_index: int = 0,
        namespace: str = "",
    ) -> Decision:
        """Sync call. A `namespace` (e.g. "latency") keeps a separate cache entry so
        latency runs make their own timed calls instead of reusing batch results."""
        params = self.params(example, temperature=temperature)
        key = self.key(example, params, sample_index, namespace)
        hit = self.cache.get(key)
        if hit is not None:
            return Decision.model_validate({**hit, "cached": True})

        call = dict(params)
        # SDK 1.x dropped `temperature` from create(); the API still honours it on
        # Haiku 4.5, so it travels in extra_body. (Batch params forward it as-is.)
        if "temperature" in call:
            call["extra_body"] = {"temperature": call.pop("temperature")}
        start = time.perf_counter()
        try:
            msg = await self.client.messages.create(**call)
        except anthropic.APIError as e:
            return Decision(
                confidence_type="verbalized",
                sample_index=sample_index,
                **failure("api_error", f"{type(e).__name__}: {e}"),  # type: ignore[arg-type]
            )
        latency_ms = (time.perf_counter() - start) * 1000
        decision = parse_message(msg.to_dict(), self.model, label_space, batch=False)
        decision = decision.model_copy(
            update={"latency_ms": latency_ms, "sample_index": sample_index}
        )
        self.cache.put(key, decision.model_dump(mode="json"))
        return decision

    # ---- batch mode -------------------------------------------------------

    def batch_request(
        self, example: Example, *, temperature: float | None = None, sample_index: int = 0
    ) -> tuple[str, dict[str, Any]]:
        """(custom_id, params) for one Message Batches request; custom_id is the cache key."""
        params = self.params(example, temperature=temperature, use_cache=True)
        return self.key(example, params, sample_index), params

    def parse_batch_result(
        self, result: dict[str, Any], label_space: tuple[str, ...], sample_index: int = 0
    ) -> Decision:
        """Parse one entry of `batches.results()` (as a dict). Latency is not measured."""
        r = result["result"]
        if r["type"] != "succeeded":
            err = r.get("error", {}).get("error", r.get("error", {}))
            return Decision(
                confidence_type="verbalized",
                sample_index=sample_index,
                **failure("api_error", f"batch {r['type']}: {err}"),  # type: ignore[arg-type]
            )
        d = parse_message(r["message"], self.model, label_space, batch=True)
        return d.model_copy(update={"sample_index": sample_index})


def parse_message(
    msg: dict[str, Any], model: ModelConfig, label_space: tuple[str, ...], *, batch: bool
) -> Decision:
    u = msg.get("usage") or {}
    usage = Usage(
        input_tokens=u.get("input_tokens") or 0,
        output_tokens=u.get("output_tokens") or 0,
        cache_creation_input_tokens=u.get("cache_creation_input_tokens") or 0,
        cache_read_input_tokens=u.get("cache_read_input_tokens") or 0,
    )
    base: dict[str, Any] = {
        "confidence_type": "verbalized",
        "usage": usage,
        "cost_usd": model.cost(usage, batch=batch),
        "model_version": msg.get("model"),
        "raw": {
            "id": msg.get("id"),
            "stop_reason": msg.get("stop_reason"),
            "content": msg.get("content"),
        },
    }
    stop = msg.get("stop_reason")
    if stop == "refusal":
        return Decision(**base, **failure("refusal", str(msg.get("stop_details"))))  # type: ignore[arg-type]
    text = "".join(b.get("text", "") for b in msg.get("content", []) if b.get("type") == "text")
    if stop == "max_tokens":
        return Decision(**base, **failure("invalid_output", f"truncated: {text}"))  # type: ignore[arg-type]
    try:
        data = json.loads(text)
        label, conf = data["label"], float(data["confidence"])
    except (ValueError, KeyError, TypeError) as e:
        return Decision(**base, **failure("invalid_output", f"{e}: {text}"))  # type: ignore[arg-type]
    if not 0.0 <= conf <= 1.0:
        return Decision(**base, **failure("invalid_output", f"confidence out of range: {conf}"))  # type: ignore[arg-type]
    return check_label(Decision(**base, label=label, confidence=conf), label_space)


def aggregate_self_consistency(samples: Sequence[Decision]) -> Decision:
    """Majority vote over k samples; confidence = agreement rate (votes / k).

    Failed samples count toward k but cast no vote, so failures lower confidence.
    Ties go to the label seen first, which is deterministic for a fixed sample order.
    """
    k = len(samples)
    votes = Counter(s.label for s in samples if s.error is None and s.label is not None)
    usage = Usage(
        input_tokens=sum(s.usage.input_tokens for s in samples),
        output_tokens=sum(s.usage.output_tokens for s in samples),
        cache_creation_input_tokens=sum(s.usage.cache_creation_input_tokens for s in samples),
        cache_read_input_tokens=sum(s.usage.cache_read_input_tokens for s in samples),
    )
    ctype: ConfidenceType = "self_consistency"
    common: dict[str, Any] = {
        "confidence_type": ctype,
        "usage": usage,
        "cost_usd": sum(s.cost_usd for s in samples),
        "model_version": next((s.model_version for s in samples if s.model_version), None),
        "probabilities": {lab: n / k for lab, n in votes.items()} if votes else None,
        "cached": all(s.cached for s in samples),
    }
    if not votes:
        return Decision(**common, **failure("invalid_output", f"all {k} samples failed"))  # type: ignore[arg-type]
    label, n = votes.most_common(1)[0]
    return Decision(**common, label=label, confidence=n / k)
