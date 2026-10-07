import asyncio
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from jev_bench.adapters.claude import ClaudeDecider, aggregate_self_consistency
from jev_bench.types import Decision, Usage

HAIKU = "claude-haiku-4-5-20251001"


def message(text=None, stop="end_turn", usage=None, model=HAIKU):
    content = [] if text is None else [{"type": "text", "text": text}]
    return {
        "id": "msg_1",
        "model": model,
        "stop_reason": stop,
        "content": content,
        "usage": usage or {"input_tokens": 400, "output_tokens": 20},
    }


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def create(self, **params):
        self.calls.append(params)
        if isinstance(self.response, Exception):
            raise self.response
        response = self.response
        return SimpleNamespace(to_dict=lambda: response)


def make(cfg, spec, cache, response, model="haiku"):
    fake = FakeMessages(response)
    client = SimpleNamespace(messages=fake)
    return ClaudeDecider(model, cfg.models[model], spec, cache, client=client), fake


def run(d, example, **kw):
    return asyncio.run(d.decide(example, example.label_space, **kw))


def test_valid_verbalized(cfg, spec, cache, example):
    d, fake = make(cfg, spec, cache, message('{"label": "negative", "confidence": 0.9}'))
    r = run(d, example)
    assert (r.label, r.confidence, r.error) == ("negative", 0.9, None)
    assert r.confidence_type == "verbalized"
    assert r.cost_usd == pytest.approx((400 * 1 + 20 * 5) / 1e6)  # standard price in sync mode
    p = fake.calls[0]
    # SDK 1.x has no `temperature` kwarg; it must travel in extra_body
    assert "temperature" not in p and p["extra_body"] == {"temperature": 0.0}
    assert "thinking" not in p and "tool_choice" not in p
    schema = p["output_config"]["format"]["schema"]
    assert schema["properties"]["label"]["enum"] == ["negative", "positive"]
    assert "Dislikes it" in p["system"][0]["text"]  # same descriptions as Jev
    assert "cache_control" not in p["system"][0]  # never cache on sync/latency calls


def test_sonnet_params_follow_docs(cfg, spec, cache, example):
    d, fake = make(cfg, spec, cache, message('{"label": "positive", "confidence": 0.6}'), "sonnet")
    run(d, example)
    p = fake.calls[0]
    assert "temperature" not in p and "extra_body" not in p  # Sonnet 5.5: default sampling
    assert p["thinking"] == {"type": "between_tools"}


@pytest.mark.parametrize(
    ("msg", "kind"),
    [
        (message(stop="refusal"), "refusal"),
        (message('{"label": "neg', stop="max_tokens"), "invalid_output"),
        (message("not json"), "invalid_output"),
        (message('{"label": "negative"}'), "invalid_output"),
        (message('{"label": "negative", "confidence": 1.7}'), "invalid_output"),
        (message('{"label": "neutral", "confidence": 0.5}'), "out_of_label_set"),
    ],
)
def test_failures_are_typed_and_kept(cfg, spec, cache, example, msg, kind):
    d, _ = make(cfg, spec, cache, msg)
    r = run(d, example)
    assert r.error == kind
    assert r.usage.input_tokens == 400  # failed calls still cost money


def test_api_error_not_cached(cfg, spec, cache, example):
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    d, fake = make(cfg, spec, cache, anthropic.APIConnectionError(request=req))
    assert run(d, example).error == "api_error"
    run(d, example)
    assert len(fake.calls) == 2


def test_cache_and_sample_index(cfg, spec, cache, example):
    d, fake = make(cfg, spec, cache, message('{"label": "negative", "confidence": 0.9}'))
    run(d, example)
    assert run(d, example).cached and len(fake.calls) == 1
    run(d, example, temperature=1.0, sample_index=1)
    run(d, example, temperature=1.0, sample_index=2)
    assert len(fake.calls) == 3
    assert fake.calls[1]["extra_body"] == {"temperature": 1.0}
    run(d, example, namespace="latency")  # latency runs never reuse accuracy results
    assert len(fake.calls) == 4


def test_batch_request_and_result(cfg, spec, cache, example):
    d, _ = make(cfg, spec, cache, None, "sonnet")
    custom_id, params = d.batch_request(example)
    assert len(custom_id) == 64
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    haiku, _ = make(cfg, spec, cache, None)
    assert haiku.batch_request(example)[1]["temperature"] == 0.0  # batch params forward it
    # batch and sync share the cache key: a batch result serves later sync reruns
    assert custom_id == d.key(example, d.params(example))
    usage = {"input_tokens": 10, "output_tokens": 20, "cache_read_input_tokens": 1000}
    msg = message(
        '{"label": "positive", "confidence": 0.8}', usage=usage, model="claude-sonnet-5-5"
    )
    res = {"custom_id": custom_id, "result": {"type": "succeeded", "message": msg}}
    r = d.parse_batch_result(res, example.label_space)
    assert r.label == "positive" and r.latency_ms is None
    assert r.cost_usd == pytest.approx((10 * 2 + 1000 * 2 * 0.1 + 20 * 10) / 1e6 * 0.5)
    expired = {"custom_id": custom_id, "result": {"type": "expired"}}
    assert d.parse_batch_result(expired, example.label_space).error == "api_error"


def _s(label, error=None):
    return Decision(
        label=label,
        confidence=0.9,
        confidence_type="verbalized",
        error=error,
        cost_usd=0.001,
        usage=Usage(input_tokens=10),
    )


def test_self_consistency_majority():
    r = aggregate_self_consistency([_s("a"), _s("b"), _s("a"), _s("a"), _s("b")])
    assert (r.label, r.confidence, r.confidence_type) == ("a", 0.6, "self_consistency")
    assert r.probabilities == {"a": 0.6, "b": 0.4}
    assert r.cost_usd == pytest.approx(0.005) and r.usage.input_tokens == 50


def test_self_consistency_failures_lower_confidence():
    samples = [_s("a"), _s("a"), _s(None, "invalid_output"), _s("a"), _s("a")]
    assert aggregate_self_consistency(samples).confidence == pytest.approx(0.8)


def test_self_consistency_all_failed():
    r = aggregate_self_consistency([_s(None, "refusal")] * 5)
    assert r.error == "invalid_output" and r.label is None
