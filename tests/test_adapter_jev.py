import asyncio
import json

import httpx2
import pytest
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from jev_bench.adapters.jev import JevDecider


def make(cfg, spec, cache, handler):
    calls = []

    def h(req):
        calls.append(json.loads(req.content))
        return handler(req)

    client = AsyncTypeSafeClient(
        api_key="test", transport=httpx2.MockTransport(h), retry=RetryPolicy(max_retries=0)
    )
    return JevDecider("jev", cfg.models["jev"], spec, cache, client=client), calls


def ok(probs, choice=None, confidence=1.0):
    choice = choice or max(probs, key=probs.get)
    answer = {"type": "choice", "choice": choice, "confidence": confidence, "probabilities": probs}
    body = {
        "model": "jev-1.13.0",
        "usage": {"input_tokens": 320, "output_tokens": 32},
        "answers": {"label": answer},
    }
    return lambda req: httpx2.Response(200, json=body)


def run(decider, example):
    return asyncio.run(decider.decide(example, example.label_space))


def test_hard_probabilities(cfg, spec, cache, example):
    d, calls = make(cfg, spec, cache, ok({"positive": 0.0, "negative": 1.0}))
    r = run(d, example)
    assert (r.label, r.confidence, r.api_confidence) == ("negative", 1.0, 1.0)
    assert r.confidence_type == "native_max_prob" and r.error is None
    assert r.model_version == "jev-1.13.0"
    assert r.usage.input_tokens == 320
    assert r.cost_usd == pytest.approx(320 * 0.042 / 1e6)  # output tokens free
    assert calls[0]["questions"]["label"]["criteria"] == spec.labels  # shared descriptions


def test_max_prob_differs_from_api_confidence(cfg, spec, cache, example):
    d, _ = make(cfg, spec, cache, ok({"positive": 0.3, "negative": 0.7}, confidence=0.12))
    r = run(d, example)
    assert r.confidence == pytest.approx(0.7) and r.api_confidence == pytest.approx(0.12)


def test_cache_hit_makes_no_call(cfg, spec, cache, example):
    d, calls = make(cfg, spec, cache, ok({"positive": 0.2, "negative": 0.8}))
    first, second = run(d, example), run(d, example)
    assert len(calls) == 1
    assert not first.cached and second.cached and second.label == first.label


def test_api_error_recorded_not_cached(cfg, spec, cache, example):
    d, calls = make(cfg, spec, cache, lambda req: httpx2.Response(500, json={"error": "boom"}))
    r = run(d, example)
    assert r.error == "api_error" and r.label is None
    run(d, example)
    assert len(calls) == 2  # retried on rerun, not served from cache


def test_out_of_label_set(cfg, spec, cache, example):
    d, _ = make(cfg, spec, cache, ok({"neutral": 1.0}))
    assert run(d, example).error == "out_of_label_set"


def test_missing_answer_is_invalid(cfg, spec, cache, example):
    body = {"model": "jev-1.13.0", "usage": {"input_tokens": 1, "output_tokens": 0}, "answers": {}}
    d, _ = make(cfg, spec, cache, lambda req: httpx2.Response(200, json=body))
    assert run(d, example).error == "invalid_output"
