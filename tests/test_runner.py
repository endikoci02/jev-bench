import asyncio
import json
from types import SimpleNamespace

import pytest

from jev_bench.adapters.claude import ClaudeDecider
from jev_bench.budget import BudgetCapReachedError, BudgetExceededError, SpendLedger
from jev_bench.runner import BatchItem, SpendTracker, run_batch, run_sync, write_raw_log
from jev_bench.types import Decision, Example

# ---- SpendTracker -----------------------------------------------------------


def test_tracker_buffers_then_flushes(tmp_path):
    led = SpendLedger(tmp_path / "l.json", 1.0)
    t = SpendTracker(led, "r", flush_every_usd=0.10)
    t.add("haiku", 0.04)
    t.add("sonnet", 0.04)
    assert led.spent == 0  # buffered
    t.add("haiku", 0.04)  # 0.12 pending >= 0.10 -> flush
    assert led.spent == pytest.approx(0.12)
    assert led.spent_by_model() == pytest.approx({"haiku": 0.08, "sonnet": 0.04})


def test_tracker_aborts_at_cap_and_records_everything(tmp_path):
    led = SpendLedger(tmp_path / "l.json", 0.10)
    t = SpendTracker(led, "r", flush_every_usd=1.0)
    t.add("haiku", 0.06)
    with pytest.raises(BudgetCapReachedError):
        t.add("sonnet", 0.05)
    assert SpendLedger(tmp_path / "l.json", 0.10).spent == pytest.approx(0.11)


# ---- run_sync ---------------------------------------------------------------


def ex(i, label="a"):
    return Example(id=f"e{i}", text=f"text {i}", label=label, label_space=("a", "b"))


class FakeDecider:
    name = "haiku"

    def __init__(self, cost=0.01, delay=0.0):
        self.cost, self.delay = cost, delay
        self.in_flight = self.max_in_flight = 0
        self.namespaces = []

    async def decide(self, example, label_space, *, namespace=""):
        self.namespaces.append(namespace)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(self.delay)
        self.in_flight -= 1
        return Decision(
            label=example.text, confidence=0.5, confidence_type="verbalized", cost_usd=self.cost
        )


def test_run_sync_order_concurrency_and_warmup():
    d = FakeDecider(delay=0.01)
    exs = [ex(i) for i in range(12)]
    out = asyncio.run(
        run_sync(d, exs, concurrency=3, tracker=None, namespace="latency", sequential_first=2)
    )
    assert [o.label for o in out] == [e.text for e in exs]
    assert d.max_in_flight == 3 and set(d.namespaces) == {"latency"}


def test_run_sync_aborts_mid_run_at_cap(tmp_path):
    led = SpendLedger(tmp_path / "l.json", 0.05)
    tracker = SpendTracker(led, "r", flush_every_usd=1.0)
    d = FakeDecider(cost=0.01)
    with pytest.raises(BudgetCapReachedError):
        asyncio.run(run_sync(d, [ex(i) for i in range(50)], concurrency=1, tracker=tracker))
    assert led.spent == pytest.approx(0.05)  # stopped at the cap, not after 50 calls


# ---- run_batch --------------------------------------------------------------


def ok_message(label):
    return {
        "id": "msg", "model": "claude-haiku-4-5-20251001", "stop_reason": "end_turn",
        "content": [{"type": "text", "text": json.dumps({"label": label, "confidence": 0.7})}],
        "usage": {"input_tokens": 1000, "output_tokens": 20},
    }  # fmt: skip


class FakeBatches:
    """Simulates the Message Batches API. `fail_ids` come back errored."""

    def __init__(self, fail_ids=()):
        self.submitted = []
        self.fail_ids = set(fail_ids)
        self.polls = 0
        self.store = {}

    def create(self, requests):
        bid = f"batch_{len(self.submitted)}"
        self.submitted.append([r["custom_id"] for r in requests])
        self.store[bid] = requests
        return SimpleNamespace(id=bid)

    def retrieve(self, bid):
        self.polls += 1
        status = "ended" if self.polls % 2 == 0 else "in_progress"
        counts = SimpleNamespace(succeeded=0, errored=0, processing=0)
        return SimpleNamespace(processing_status=status, request_counts=counts)

    def results(self, bid):
        for r in self.store[bid]:
            cid = r["custom_id"]
            if cid in self.fail_ids:
                body = {"custom_id": cid, "result": {"type": "errored", "error": {"type": "x"}}}
            else:
                text = r["params"]["messages"][0]["content"]
                body = {"custom_id": cid, "result": {"type": "succeeded", "message": ok_message(
                    "a" if "0" in text else "b")}}  # fmt: skip
            yield SimpleNamespace(to_dict=lambda b=body: b)


def setup(cfg, spec, cache, tmp_path, cap=4.0, fail_ids=()):
    decider = ClaudeDecider("haiku", cfg.models["haiku"], spec, cache)
    fake = FakeBatches(fail_ids)
    client = SimpleNamespace(messages=SimpleNamespace(batches=fake))
    led = SpendLedger(tmp_path / "ledger.json", cap)
    tracker = SpendTracker(led, "r")
    kw = dict(
        client=client, tracker=tracker, state_path=tmp_path / "state.json",
        cost_per_call=0.001, chunk_size=2, poll_seconds=0, log=lambda m: None,
        sleep=lambda s: None,
    )  # fmt: skip
    return decider, fake, led, kw


def items(n):
    return [BatchItem(Example(id=f"e{i}", text=f"t{i}", label="a", label_space=("a", "b")))
            for i in range(n)]  # fmt: skip


def test_batch_end_to_end_chunks_cost_and_cache(cfg, spec, cache, tmp_path):
    decider, fake, led, kw = setup(cfg, spec, cache, tmp_path)
    out = run_batch(decider, items(5), **kw)
    assert len(fake.submitted) == 3 and [len(c) for c in fake.submitted] == [2, 2, 1]
    assert [d.label for d in out] == ["a", "b", "b", "b", "b"]
    per_call = (1000 * 1 + 20 * 5) / 1e6 * 0.5
    assert led.spent == pytest.approx(5 * per_call)
    assert not (tmp_path / "state.json").exists()
    # rerun: everything cached, nothing submitted, $0
    out2 = run_batch(decider, items(5), **kw)
    assert len(fake.submitted) == 3 and all(d.cached for d in out2)
    assert led.spent == pytest.approx(5 * per_call)


def test_batch_errors_kept_not_cached_then_retried(cfg, spec, cache, tmp_path):
    decider, fake, _, kw = setup(cfg, spec, cache, tmp_path)
    first_id = decider.batch_request(items(1)[0].example)[0]
    fake.fail_ids = {first_id}
    out = run_batch(decider, items(3), **kw)
    assert out[0].error == "api_error" and out[1].error is None
    fake.fail_ids = set()
    out2 = run_batch(decider, items(3), **kw)
    assert fake.submitted[-1] == [first_id]  # only the failed one is resubmitted
    assert out2[0].error is None


def test_batch_resume_after_interruption(cfg, spec, cache, tmp_path):
    decider, fake, led, kw = setup(cfg, spec, cache, tmp_path)
    # simulate a crash after submitting: state file names a batch not yet collected
    its = items(2)
    reqs = [decider.batch_request(i.example) for i in its]
    fake.create([{"custom_id": c, "params": p} for c, p in reqs])
    (tmp_path / "state.json").write_text(
        json.dumps({"chunks": [{"batch_id": "batch_0", "custom_ids": [c for c, _ in reqs]}]})
    )
    out = run_batch(decider, its, **kw)
    assert len(fake.submitted) == 1  # resumed by polling, not resubmitted
    assert all(d.error is None for d in out) and led.spent > 0


def test_batch_chunk_refused_when_over_remaining_budget(cfg, spec, cache, tmp_path):
    decider, fake, _, kw = setup(cfg, spec, cache, tmp_path, cap=0.0015)
    with pytest.raises(BudgetExceededError):
        run_batch(decider, items(4), **kw)  # chunk of 2 x $0.001 > $0.0015
    assert fake.submitted == []


def test_raw_log_marks_warmup(tmp_path):
    d = Decision(label="a", confidence=1.0, confidence_type="native_max_prob")
    p = write_raw_log(
        run_id="r", dataset="sst2", model="jev", mode="latency", examples=[ex(0), ex(1)],
        decisions=[d, d], warmup=1, path=tmp_path / "x.jsonl",
    )  # fmt: skip
    lines = [json.loads(line) for line in p.read_text().splitlines()]
    assert [ln["warmup"] for ln in lines] == [True, False]
    assert lines[0]["gold"] == "a" and lines[0]["decision"]["label"] == "a"
