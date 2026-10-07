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


POP = ["a"] * 700 + ["b"] * 200 + ["c"] * 100


@pytest.mark.parametrize("n", [5, 20, 100, 200, 537])
@pytest.mark.parametrize("seed", [0, 42])
def test_every_prefix_within_one_of_exact_share(n, seed):
    counts = Counter(POP[i] for i in stratified_sample(POP, n, seed))
    for cls, size in Counter(POP).items():
        assert abs(counts[cls] - n * size / len(POP)) < 1 + 1e-9


def test_samples_are_nested_and_deterministic():
    dev, full = stratified_sample(POP, 20, 42), stratified_sample(POP, 200, 42)
    assert full[:20] == dev  # dev runs are reused by the full run
    assert len(set(full)) == 200
    assert full == stratified_sample(POP, 200, 42)
    assert full != stratified_sample(POP, 200, 7)


def test_rare_class_appears_once_its_share_reaches_one():
    pop = ["common"] * 990 + ["rare"] * 10  # share 1%
    assert "rare" in {pop[i] for i in stratified_sample(pop, 100, 0)}


def test_limit_above_population_returns_everything():
    assert sorted(stratified_sample(["a", "b"], 10, seed=0)) == [0, 1]
