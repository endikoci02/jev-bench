from collections import Counter

import pytest

from jev_bench.datasets import load_label_spec, stratified_sample


@pytest.mark.parametrize(
    ("name", "n"), [("sst2", 2), ("banking77", 77), ("clinc150", 151), ("offensive", 2)]
)
def test_label_specs_complete(name, n):
    spec = load_label_spec(name)
    assert len(spec.labels) == n
    assert all(desc.strip() for desc in spec.labels.values())


def test_clinc_has_oos_and_yaml_bool_trap_avoided():
    labels = load_label_spec("clinc150").labels
    assert {"oos", "yes", "no"} <= set(labels)  # unquoted yes/no would parse as booleans


def test_banking77_keeps_odd_hf_label_names():
    labels = load_label_spec("banking77").labels
    assert "Refund_not_showing_up" in labels and "reverted_card_payment?" in labels


def test_stratified_sample_proportions_and_determinism():
    pop = ["a"] * 700 + ["b"] * 200 + ["c"] * 100
    idx = stratified_sample(pop, 100, seed=42)
    assert len(idx) == 100 == len(set(idx))
    assert Counter(pop[i] for i in idx) == {"a": 70, "b": 20, "c": 10}
    assert idx == stratified_sample(pop, 100, seed=42)
    assert idx != stratified_sample(pop, 100, seed=7)


def test_stratified_sample_largest_remainder():
    pop = ["a"] * 5 + ["b"] * 3 + ["c"] * 2  # n=5 -> exact quotas 2.5, 1.5, 1.0
    counts = Counter(pop[i] for i in stratified_sample(pop, 5, seed=0))
    assert sum(counts.values()) == 5 and counts["c"] == 1


def test_stratified_sample_limit_above_population():
    assert stratified_sample(["a", "b"], 10, seed=0) == [0, 1]
