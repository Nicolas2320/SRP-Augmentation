import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100


def group_indices_by_class(targets):
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_to_indices[int(label)].append(idx)
    return class_to_indices


def sample_per_class(class_to_indices, n_per_class, seed):
    rng = random.Random(seed)
    selected = []

    for label, indices in class_to_indices.items():
        indices_copy = list(indices)
        rng.shuffle(indices_copy)

        if len(indices_copy) < n_per_class:
            raise ValueError(
                f"Class {label} has only {len(indices_copy)} samples, "
                f"but requested {n_per_class}."
            )

        selected.extend(indices_copy[:n_per_class])

    selected = sorted(selected)
    return selected


def create_validation_and_pool_indices(targets, val_per_class, seed):
    rng = random.Random(seed)
    class_to_indices = group_indices_by_class(targets)

    val_indices = []
    train_pool_indices = []

    for label, indices in class_to_indices.items():
        indices_copy = list(indices)
        rng.shuffle(indices_copy)

        val_part = indices_copy[:val_per_class]
        train_part = indices_copy[val_per_class:]

        val_indices.extend(val_part)
        train_pool_indices.extend(train_part)

    return sorted(val_indices), sorted(train_pool_indices)


def create_kshot_from_pool(targets, train_pool_indices, k, seed):
    pool_class_to_indices = defaultdict(list)

    for idx in train_pool_indices:
        label = int(targets[idx])
        pool_class_to_indices[label].append(idx)

    return sample_per_class(pool_class_to_indices, n_per_class=k, seed=seed)


def max_k_per_class(targets, train_pool_indices):
    """Return the largest balanced per-class subset available in the pool."""
    pool_class_to_indices = defaultdict(list)

    for idx in train_pool_indices:
        label = int(targets[idx])
        pool_class_to_indices[label].append(idx)

    if not pool_class_to_indices:
        raise ValueError("Training pool must not be empty.")
    return min(len(indices) for indices in pool_class_to_indices.values())


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    root = Path("data/raw")
    output_root = Path("data/splits")

    datasets = {
        "cifar10": {
            "class": CIFAR10,
            "val_per_class": 500,
            "k_values": [5, 10, 20, 50, 100, 200, 300],
        },
        "cifar100": {
            "class": CIFAR100,
            "val_per_class": 50,
            "k_values": [5, 10, 20, 50, 100, 200, 300],
        },
    }

    validation_seed = 123
    subset_seeds = [0, 1, 2]

    for dataset_name, cfg in datasets.items():
        print(f"Processing {dataset_name}...")

        dataset = cfg["class"](
            root=root,
            train=True,
            download=True,
        )

        targets = dataset.targets

        val_indices, train_pool_indices = create_validation_and_pool_indices(
            targets=targets,
            val_per_class=cfg["val_per_class"],
            seed=validation_seed,
        )

        split_info = {
            "dataset": dataset_name,
            "validation_seed": validation_seed,
            "val_per_class": cfg["val_per_class"],
            "num_val": len(val_indices),
            "num_train_pool": len(train_pool_indices),
            "val_indices": val_indices,
            "train_pool_indices": train_pool_indices,
        }

        save_json(
            output_root / dataset_name / "fixed_validation_split.json",
            split_info,
        )

        max_k = max_k_per_class(targets, train_pool_indices)
        k_values = [*cfg["k_values"], max_k]

        for k in k_values:
            # The maximum subset contains the entire post-validation pool, so
            # subset seeds would all produce the same indices. Save it once.
            seeds_for_k = [0] if k == max_k else subset_seeds
            for seed in seeds_for_k:
                train_indices = create_kshot_from_pool(
                    targets=targets,
                    train_pool_indices=train_pool_indices,
                    k=k,
                    seed=seed,
                )

                subset_info = {
                    "dataset": dataset_name,
                    "k": k,
                    "subset_seed": seed,
                    "validation_seed": validation_seed,
                    "num_train": len(train_indices),
                    "train_indices": train_indices,
                }
                if k == max_k:
                    subset_info["is_max_train_pool"] = True

                save_json(
                    output_root / dataset_name / f"k{k}_seed{seed}.json",
                    subset_info,
                )

                print(
                    f"Saved {dataset_name} | k={k} | seed={seed} | "
                    f"n={len(train_indices)}"
                )


if __name__ == "__main__":
    main()
