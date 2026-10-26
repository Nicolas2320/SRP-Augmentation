"""Audit committed balanced splits before cluster training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from torchvision.datasets import CIFAR100

try:
    from .make_splits import create_kshot_from_pool
    from .tiny_imagenet import TinyImageNet
except ImportError:  # Support `python src/data/audit_splits.py`.
    from make_splits import create_kshot_from_pool
    from tiny_imagenet import TinyImageNet


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def audit_fixed_split(
    fixed: dict[str, Any],
    targets: list[int],
    dataset_name: str,
) -> tuple[set[int], set[int]]:
    validation_list = [int(index) for index in fixed["val_indices"]]
    train_pool_list = [int(index) for index in fixed["train_pool_indices"]]
    validation = set(validation_list)
    train_pool = set(train_pool_list)
    all_indices = set(range(len(targets)))

    if fixed.get("dataset") != dataset_name:
        raise AssertionError("fixed split dataset does not match the requested dataset")
    if validation_list != sorted(validation_list) or train_pool_list != sorted(train_pool_list):
        raise AssertionError("fixed split indices are not sorted")
    if len(validation) != len(validation_list) or len(train_pool) != len(train_pool_list):
        raise AssertionError("fixed split contains duplicate indices")
    if validation.intersection(train_pool):
        raise AssertionError("fixed training and validation pools overlap")
    if validation.union(train_pool) != all_indices:
        raise AssertionError("fixed split does not partition the official training set exactly")
    if len(validation) != int(fixed["num_val"]):
        raise AssertionError("fixed split num_val mismatch")
    if len(train_pool) != int(fixed["num_train_pool"]):
        raise AssertionError("fixed split num_train_pool mismatch")

    val_per_class = int(fixed["val_per_class"])
    validation_counts = Counter(int(targets[index]) for index in validation)
    pool_counts = Counter(int(targets[index]) for index in train_pool)
    class_counts = Counter(int(label) for label in targets)
    if set(validation_counts) != set(class_counts) or set(validation_counts.values()) != {val_per_class}:
        raise AssertionError(f"fixed validation must contain exactly {val_per_class} images per class")
    expected_pool_counts = {
        label: count - val_per_class for label, count in class_counts.items()
    }
    if dict(pool_counts) != expected_pool_counts:
        raise AssertionError("fixed training pool has incorrect per-class counts")

    if dataset_name == "tinyimagenet":
        expected_policy = {
            "class_mapping_source": "wnids.txt",
            "validation_source": "reserved_from_official_train",
            "test_source": "official_labeled_val",
            "official_unlabeled_test_used": False,
        }
        for key, expected in expected_policy.items():
            if fixed.get(key) != expected:
                raise AssertionError(f"Tiny ImageNet fixed split has invalid {key}")

    return train_pool, validation


def audit_split(
    split: dict[str, Any],
    targets: list[int],
    train_pool: set[int],
    validation: set[int],
    num_classes: int,
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
    if len(counts) != num_classes or set(counts.values()) != {k}:
        raise AssertionError(f"k={k} seed={seed}: expected exactly {k} examples per class")

    regenerated = create_kshot_from_pool(targets, sorted(train_pool), k=k, seed=seed)
    if indices != regenerated:
        raise AssertionError(f"k={k} seed={seed}: deterministic regeneration mismatch")
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--split-root", type=Path, default=Path("data/splits"))
    parser.add_argument(
        "--dataset", choices=["cifar100", "tinyimagenet"], default="cifar100"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "tinyimagenet":
        dataset = TinyImageNet(root=args.data_root, split="train")
        official_test = TinyImageNet(root=args.data_root, split="val")
        if len(dataset.classes) != 200 or len(official_test) != 10_000:
            raise AssertionError("Tiny ImageNet must contain 200 classes and 10,000 labeled test images")
        if len(Counter(official_test.targets)) != 200 or set(Counter(official_test.targets).values()) != {50}:
            raise AssertionError("Tiny ImageNet official validation/test partition must have 50 images per class")
    else:
        dataset = CIFAR100(root=args.data_root, train=True, download=False)
    split_dir = args.split_root / args.dataset
    fixed = load_json(split_dir / "fixed_validation_split.json")
    train_pool, validation = audit_fixed_split(fixed, dataset.targets, args.dataset)

    split_paths = sorted(split_dir.glob("k*_seed*.json"))
    if args.dataset == "tinyimagenet":
        expected_names = {
            *(f"k{k}_seed{seed}.json" for k in (20, 100) for seed in (0, 1, 2)),
            "k450_seed0.json",
        }
        actual_names = {path.name for path in split_paths}
        if actual_names != expected_names:
            raise AssertionError(
                "Tiny ImageNet requires exactly the seven frozen split files; "
                f"missing={sorted(expected_names - actual_names)}, "
                f"unexpected={sorted(actual_names - expected_names)}"
            )

    by_seed: dict[int, list[tuple[int, set[int]]]] = {}
    audited = 0
    for path in split_paths:
        split = load_json(path)
        indices = audit_split(
            split,
            dataset.targets,
            train_pool,
            validation,
            num_classes=len(set(dataset.targets)),
        )
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

    print(
        f"Audited {audited} {args.dataset} split files: "
        "balance, determinism, nesting, and leakage OK"
    )


if __name__ == "__main__":
    main()
