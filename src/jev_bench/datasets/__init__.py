"""Dataset loading: Hugging Face splits -> seeded, stratified `Example` samples."""

from __future__ import annotations

import os
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel

from jev_bench.config import BenchConfig, DatasetConfig
from jev_bench.types import Example

LABELS_DIR = Path(__file__).parent / "labels"


class LabelSpec(BaseModel):
    """Question + option descriptions, shared verbatim by Jev and Claude."""

    instructions: str
    labels: dict[str, str]

    @property
    def label_space(self) -> tuple[str, ...]:
        return tuple(self.labels)


def load_label_spec(dataset: str) -> LabelSpec:
    with (LABELS_DIR / f"{dataset}.yaml").open(encoding="utf-8-sig") as f:
        return LabelSpec.model_validate(yaml.safe_load(f))


def stratified_sample(labels: Sequence[str], n: int, seed: int) -> list[int]:
    """Indices of a size-n sample whose class mix matches the population.

    Per-class quotas use largest-remainder rounding; ties are broken by a
    seeded shuffle so the result is deterministic for a given seed.
    """
    if n >= len(labels):
        return list(range(len(labels)))
    rng = random.Random(seed)
    by_class: dict[str, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by_class[lab].append(i)
    classes = sorted(by_class)
    exact = {c: n * len(by_class[c]) / len(labels) for c in classes}
    quota = {c: int(exact[c]) for c in classes}
    leftover = n - sum(quota.values())
    order = classes[:]
    rng.shuffle(order)
    order.sort(key=lambda c: exact[c] - quota[c], reverse=True)  # stable: shuffle breaks ties
    for c in order[:leftover]:
        quota[c] += 1
    picked: list[int] = []
    for c in classes:
        picked.extend(rng.sample(by_class[c], quota[c]))
    return sorted(picked)


def load_examples(name: str, cfg: BenchConfig, limit: int) -> list[Example]:
    """Load a fixed, seeded, stratified sample. Same call -> same examples."""
    os.environ.setdefault("HF_HOME", str(cfg.resolve(cfg.hf_cache_dir)))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    import datasets  # imported late so HF_HOME applies

    dcfg: DatasetConfig = cfg.datasets[name]
    spec = load_label_spec(name)
    ds = datasets.load_dataset(dcfg.hf_path, dcfg.hf_config, split=dcfg.split)
    names: list[str] = ds.features[dcfg.label_field].names
    if set(names) != set(spec.labels):
        missing = set(names) ^ set(spec.labels)
        raise ValueError(f"{name}: label descriptions do not match dataset labels: {missing}")

    gold = [names[i] for i in ds[dcfg.label_field]]
    idx = stratified_sample(gold, limit, cfg.run.seed)
    texts = ds[dcfg.text_field]
    return [
        Example(
            id=f"{name}-{dcfg.split}-{i}",
            text=texts[i],
            label=gold[i],
            label_space=spec.label_space,
        )
        for i in idx
    ]
