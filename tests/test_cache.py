from pathlib import Path

from jev_bench.cache import DiskCache, cache_key

BASE = dict(
    provider="anthropic", model="m", payload={"text": "hi", "labels": ["a", "b"]}, params={"t": 0}
)


def test_key_is_stable_and_order_independent():
    a = cache_key(**BASE)
    b = cache_key(
        provider="anthropic",
        model="m",
        params={"t": 0},
        payload={"labels": ["a", "b"], "text": "hi"},
    )
    assert a == b and len(a) == 64


def test_key_changes_with_every_component():
    base = cache_key(**BASE)
    variants = [
        {**BASE, "provider": "typesafe"},
        {**BASE, "model": "m2"},
        {**BASE, "payload": {"text": "hi!", "labels": ["a", "b"]}},
        {**BASE, "params": {"t": 1}},
        {**BASE, "sample_index": 1},
    ]
    keys = {cache_key(**v) for v in variants}
    assert base not in keys and len(keys) == len(variants)


def test_miss_then_hit_roundtrip(tmp_path: Path):
    c = DiskCache(tmp_path)
    k = cache_key(**BASE)
    assert c.get(k) is None and k not in c
    c.put(k, {"label": "a", "usage": {"input_tokens": 5}})
    assert k in c
    assert DiskCache(tmp_path).get(k) == {"label": "a", "usage": {"input_tokens": 5}}


def test_no_temp_files_left(tmp_path: Path):
    c = DiskCache(tmp_path)
    c.put("ab" + "0" * 62, {"x": 1})
    assert not list(tmp_path.rglob(".tmp-*"))
