import json
from pathlib import Path

import pytest

from jev_bench.budget import BudgetCapReachedError, BudgetExceededError, SpendLedger


def ledger(tmp_path: Path, cap: float = 1.0) -> SpendLedger:
    return SpendLedger(tmp_path / "ledger.json", cap)


def test_fresh_ledger(tmp_path):
    led = ledger(tmp_path)
    assert led.spent == 0 and led.remaining == 1.0


def test_record_persists_across_instances(tmp_path):
    led = ledger(tmp_path)
    led.record(run_id="r1", model="haiku", cost_usd=0.25, calls=10)
    led.record(run_id="r1", model="sonnet", cost_usd=0.10, calls=5)
    again = ledger(tmp_path)
    assert again.spent == pytest.approx(0.35)
    assert again.spent_by_model() == pytest.approx({"haiku": 0.25, "sonnet": 0.10})


def test_refuses_estimate_over_remaining(tmp_path):
    led = ledger(tmp_path)
    led.record(run_id="r", model="haiku", cost_usd=0.8, calls=1)
    led.check_estimate(0.2)  # exactly fits
    with pytest.raises(BudgetExceededError):
        led.check_estimate(0.21)


def test_aborts_when_cap_reached_but_still_records(tmp_path):
    led = ledger(tmp_path)
    led.record(run_id="r", model="haiku", cost_usd=0.9, calls=1)
    with pytest.raises(BudgetCapReachedError):
        led.record(run_id="r", model="haiku", cost_usd=0.2, calls=1)
    # real spend is never lost, even on the call that crossed the cap
    assert ledger(tmp_path).spent == pytest.approx(1.1)
    assert ledger(tmp_path).remaining == 0.0
    with pytest.raises(BudgetCapReachedError):
        ledger(tmp_path).assert_under_cap()


def test_cap_in_file_cannot_raise_limit(tmp_path):
    led = ledger(tmp_path, cap=1.0)
    led.record(run_id="r", model="haiku", cost_usd=0.5, calls=1)
    p = tmp_path / "ledger.json"
    data = json.loads(p.read_text())
    data["cap_usd"] = 100.0
    p.write_text(json.dumps(data))
    assert ledger(tmp_path, cap=1.0).remaining == pytest.approx(0.5)


def test_negative_cost_rejected(tmp_path):
    with pytest.raises(ValueError):
        ledger(tmp_path).record(run_id="r", model="haiku", cost_usd=-1, calls=1)
