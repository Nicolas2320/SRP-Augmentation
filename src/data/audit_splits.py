"""Audit committed CIFAR-100 splits before cluster training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from torchvision.datasets import CIFAR100

from make_splits import create_kshot_from_pool


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def audit_split(
    split: dict[str, Any],
    targets: list[int],
    train_pool: set[int],
    validation: set[int],
) -> list[int]:
    k = int(split["k"])
    seed = int(split["subset_seed"])
    indices = [int(index) for index in split["train_indices"]]
    if indices != sorted(indices):
        raise AssertionError(f"k={k} seed={seed}: indices are not sorted")
    if len(indices) != len(set(indices)):
        raise AssertionError(f"k={k} seed={seed}: duplicate indices")
    if len(indices) != int(split["num_train"]):
        raise AssertionError(f"k={k} seed={seed}: num_train mismatch")
    if not set(indices).issubset(train_pool):
        raise AssertionError(f"k={k} seed={seed}: index outside fixed training pool")
    if set(indices).intersection(validation):
        raise AssertionError(f"k={k} seed={seed}: validation overlap")

    counts = Counter(int(targets[index]) for index in indices)
    if len(counts) != 100 or set(counts.values()) != {k}:
        raise AssertionError(f"k={k} seed={seed}: expected exactly {k} examples per class")

    regenerated = create_kshot_from_pool(targets, sorted(train_pool), k=k, seed=seed)
    if indices != regenerated:
        raise AssertionError(f"k={k} seed={seed}: deterministic regeneration mismatch")
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--split-root", type=Path, default=Path("data/splits"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = CIFAR100(root=args.data_root, train=True, download=False)
    split_dir = args.split_root / "cifar100"
    fixed = load_json(split_dir / "fixed_validation_split.json")
    train_pool = {int(index) for index in fixed["train_pool_indices"]}
    validation = {int(index) for index in fixed["val_indices"]}
    if train_pool.intersection(validation):
        raise AssertionError("fixed training and validation pools overlap")

    by_seed: dict[int, list[tuple[int, set[int]]]] = {}
    audited = 0
    for path in sorted(split_dir.glob("k*_seed*.json")):
        split = load_json(path)
        indices = audit_split(split, dataset.targets, train_pool, validation)
        seed = int(split["subset_seed"])
        by_seed.setdefault(seed, []).append((int(split["k"]), set(indices)))
        audited += 1
        print(f"OK {path.name}: n={len(indices)}")

    for seed, entries in by_seed.items():
        entries.sort()
        for (smaller_k, smaller), (larger_k, larger) in zip(entries, entries[1:]):
            if not smaller.issubset(larger):
                raise AssertionError(
                    f"seed={seed}: k={smaller_k} is not nested inside k={larger_k}"
                )

    print(f"Audited {audited} CIFAR-100 split files: balance, determinism, nesting, and leakage OK")


if __name__ == "__main__":
    main()
