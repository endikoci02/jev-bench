import math

import numpy as np
import pytest

from jev_bench import metrics as M


def arr(*v):
    return np.array(v, dtype=float)


def b(*v):
    return np.array(v, dtype=bool)


# ---- hand-computed ECE / MCE / Brier ---------------------------------------


def test_ece_hand_computed():
    # 15 bins of width 1/15:
    # bin 13 = (0.8667, 0.9333]: {0.9 correct, 0.9 wrong} -> acc 0.5, conf 0.9, gap 0.4
    # bin 14 = (0.9333, 1.0]:   {0.95 correct}            -> acc 1.0, conf 0.95, gap 0.05
    # bin 4  = (0.2667, 0.3333]: {0.3 wrong}              -> acc 0.0, conf 0.3, gap 0.3
    conf, correct = arr(0.9, 0.9, 0.95, 0.3), b(1, 0, 1, 0)
    assert M.ece(conf, correct) == pytest.approx((2 * 0.4 + 1 * 0.05 + 1 * 0.3) / 4)
    assert M.mce(conf, correct) == pytest.approx(0.4)


def test_brier_hand_computed():
    conf, correct = arr(0.9, 0.9, 0.95, 0.3), b(1, 0, 1, 0)
    expected = (0.1**2 + 0.9**2 + 0.05**2 + 0.3**2) / 4
    assert M.brier(conf, correct) == pytest.approx(expected)


def test_perfect_calibration_is_zero():
    conf = arr(*[0.5] * 4, *[1.0] * 3)
    correct = b(1, 0, 1, 0, 1, 1, 1)
    assert M.ece(conf, correct) == pytest.approx(0.0)


# ---- all-hard edge cases ------------------------------------------------------


def test_all_hard_all_correct():
    conf, correct = arr(1.0, 1.0, 1.0), b(1, 1, 1)
    assert M.ece(conf, correct) == 0.0 and M.brier(conf, correct) == 0.0
    assert M.hard_share(conf) == 1.0
    assert math.isnan(M.auroc(conf, correct))  # no wrong answers: undefined


def test_all_hard_some_wrong():
    # 4 predictions at 1.0, 3 right -> one bin, acc 0.75: ECE = MCE = 0.25
    conf, correct = arr(1.0, 1.0, 1.0, 1.0), b(1, 1, 1, 0)
    assert M.ece(conf, correct) == pytest.approx(0.25)
    assert M.mce(conf, correct) == pytest.approx(0.25)
    assert M.brier(conf, correct) == pytest.approx(0.25)  # one (1-0)^2 over 4
    assert M.auroc(conf, correct) == pytest.approx(0.5)  # all ties
    assert M.confident_wrong_rate(conf, correct, 0.9) == pytest.approx(0.25)
    curve = M.selective_curve(conf, correct)
    assert list(curve.coverage) == [1.0]  # ties kept together: no lower coverage


def test_hard_zero_and_one_bins():
    # confidence exactly 0 goes to the first bin, exactly 1 to the last
    idx = M.bin_index(arr(0.0, 1e-9, 1 / 15, 1.0), 15)
    assert list(idx) == [0, 0, 0, 14]
    bins = M.reliability_bins(arr(0.0, 1.0), b(0, 1))
    assert bins.count[0] == 1 and bins.count[14] == 1 and bins.count.sum() == 2


def test_hard_share_threshold():
    assert M.hard_share(arr(0.99, 0.989, 1.0, 0.5)) == pytest.approx(0.5)


# ---- AUROC / selective / misc ------------------------------------------------


def test_auroc_perfect_and_inverted():
    correct = b(1, 1, 0, 0)
    assert M.auroc(arr(0.9, 0.8, 0.3, 0.2), correct) == 1.0
    assert M.auroc(arr(0.2, 0.3, 0.8, 0.9), correct) == 0.0


def test_auroc_ties_count_half():
    # pairs (pos, neg): (0.9,0.5)=1, (0.9,0.9)=0.5, (0.5,0.5)=0.5, (0.5,0.9)=0 -> 2/4
    assert M.auroc(arr(0.9, 0.5, 0.5, 0.9), b(1, 1, 0, 0)) == pytest.approx(0.5)


def test_selective_curve():
    curve = M.selective_curve(arr(0.9, 0.8, 0.8, 0.4), b(1, 1, 0, 0))
    assert list(curve.threshold) == [0.9, 0.8, 0.4]
    assert list(curve.coverage) == [0.25, 0.75, 1.0]
    assert curve.accuracy == pytest.approx([1.0, 2 / 3, 0.5])


def test_confident_wrong_rate_is_share_of_all():
    conf, correct = arr(0.95, 0.85, 0.95, 0.5), b(0, 0, 1, 0)
    assert M.confident_wrong_rate(conf, correct, 0.9) == pytest.approx(0.25)
    assert M.confident_wrong_rate(conf, correct, 0.8) == pytest.approx(0.5)


def test_macro_f1_failures_are_false_negatives():
    gold = ["a", "a", "b", "b"]
    pred = ["a", None, "b", "a"]
    # a: tp1 fp1 fn1 -> 0.5; b: tp1 fp0 fn1 -> 2/3
    assert M.macro_f1(gold, pred, ["a", "b", "c"]) == pytest.approx((0.5 + 2 / 3) / 2)


def test_latency_percentiles():
    p = M.latency_percentiles(list(range(1, 101)))
    assert p["p50"] == pytest.approx(50.5) and p["p99"] == pytest.approx(99.01)
    assert math.isnan(M.latency_percentiles([])["p50"])


def test_bootstrap_ci_brackets_point_estimate():
    correct = np.array([1] * 80 + [0] * 20, dtype=bool)
    lo, hi = M.bootstrap_ci(lambda i: M.accuracy(correct[i]), correct.size, seed=1)
    assert lo < 0.8 < hi and lo > 0.65 and hi < 0.9


def test_empty_inputs_are_nan_not_crashes():
    e = np.array([], dtype=float)
    eb = np.array([], dtype=bool)
    for f in (M.ece, M.mce, M.brier, M.auroc):
        assert math.isnan(f(e, eb))
    assert math.isnan(M.hard_share(e))
