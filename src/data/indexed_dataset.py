"""Dataset wrappers that preserve original indices for guided mixing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Dataset


PairSampling = Literal["uniform", "weighted"]
NeighborMode = Literal["class_aware", "class_agnostic"]


class IndexedDataset(Dataset):
    """Wrap a full dataset and return samples from selected global indices."""

    def __init__(self, base_dataset: Dataset, indices: list[int] | tuple[int, ...]):
        self.base_dataset = base_dataset
        self.indices = [int(index) for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        global_index = self.indices[position]
        image, label = self.base_dataset[global_index]
        return image, label, global_index


class GuidedPairDataset(Dataset):
    """Return an anchor sample and one saved nearest-neighbor partner.

    The dataset indexes anchors by the provided k-shot ``train_indices`` while
    sampling partners from the precomputed neighbor table built over that full
    subset. Sampling is deterministic for a given ``seed`` and ``epoch`` and
    does not depend on DataLoader access order.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        train_indices: list[int] | tuple[int, ...],
        neighbor_index: dict[str, Any] | str | Path,
        pair_sampling: PairSampling = "uniform",
        mode: NeighborMode = "class_aware",
        seed: int = 0,
        anchor_mix_probs: dict[int, float] | None = None,
        dynamic_neighbor_pool: bool = False,
        easy_neighbor_pool_size: int = 3,
        hard_neighbor_pool_size: int = 10,
        difficulty_threshold: float = 0.8,
        top_rank_only: bool = False,
    ):
        if pair_sampling not in {"uniform", "weighted"}:
            raise ValueError(f"Unsupported pair_sampling: {pair_sampling}")
        if mode not in {"class_aware", "class_agnostic"}:
            raise ValueError(f"Unsupported mode: {mode}")

        self.base_dataset = base_dataset
        self.train_indices = [int(index) for index in train_indices]
        self.train_index_set = set(self.train_indices)
        self.pair_sampling = pair_sampling
        self.mode = mode
        self.seed = int(seed)
        self.epoch = 0
        self.anchor_mix_probs = self._build_anchor_mix_prob_tensor(anchor_mix_probs)
        self.dynamic_neighbor_pool = bool(dynamic_neighbor_pool)
        self.easy_neighbor_pool_size = max(1, int(easy_neighbor_pool_size))
        self.hard_neighbor_pool_size = max(1, int(hard_neighbor_pool_size))
        self.difficulty_threshold = float(difficulty_threshold)
        self.top_rank_only = bool(top_rank_only)

        payload = _load_neighbor_index(neighbor_index)
        payload_mode = payload.get("mode")
        if payload_mode is not None and payload_mode != mode:
            raise ValueError(f"Neighbor index mode is {payload_mode!r}, expected {mode!r}")

        original_indices = _as_long_tensor(payload["original_indices"], "original_indices")
        neighbor_indices = _as_long_tensor(payload["neighbor_indices"], "neighbor_indices")
        if neighbor_indices.ndim != 2:
            raise ValueError("neighbor_indices must be a 2D tensor or nested list")
        if original_indices.ndim != 1:
            raise ValueError("original_indices must be a 1D tensor or list")
        if original_indices.numel() != neighbor_indices.shape[0]:
            raise ValueError("original_indices and neighbor_indices must have matching rows")
        if neighbor_indices.shape[1] < 1:
            raise ValueError("neighbor_indices must contain at least one neighbor per sample")

        self.original_indices = original_indices
        self.neighbor_indices = neighbor_indices
        self.neighbor_rank_start = int(payload.get("neighbor_rank_start", 1))
        self.row_by_global_index = _build_index_lookup(original_indices)

        missing = sorted(index for index in self.train_indices if index not in self.row_by_global_index)
        if missing:
            raise ValueError(f"Neighbor index is missing train indices: {missing[:10]}")

        outside_subset = sorted(
            {
                int(neighbor)
                for index in self.train_indices
                for neighbor in self.neighbor_indices[self.row_by_global_index[index]].tolist()
                if int(neighbor) not in self.train_index_set
            }
        )
        if outside_subset:
            raise ValueError(f"Neighbor index includes partners outside train_indices: {outside_subset[:10]}")

        if pair_sampling == "weighted":
            if "similarities" not in payload:
                raise ValueError("Weighted pair sampling requires similarities in neighbor_index")
            similarities = torch.as_tensor(payload["similarities"], dtype=torch.float32)
            if similarities.shape != neighbor_indices.shape:
                raise ValueError("similarities must have the same shape as neighbor_indices")
            self.weights = torch.softmax(similarities, dim=1)
        else:
            self.weights = None

        if self.dynamic_neighbor_pool:
            self._validate_neighbor_pool_settings()

    def __len__(self) -> int:
        return len(self.train_indices)

    def set_epoch(self, epoch: int) -> None:
        """Change deterministic partner choices between epochs."""

        self.epoch = int(epoch)

    def sampled_neighbor_rank_counts(self) -> torch.Tensor:
        """Count selected neighbor ranks for the current epoch.

        Counts are zero-indexed internally: element 0 is the nearest neighbor,
        element 1 is the second nearest neighbor, and so on.
        """

        num_neighbors = int(self.neighbor_indices.shape[1])
        counts = torch.zeros(num_neighbors, dtype=torch.long)
        for position, idx_i in enumerate(self.train_indices):
            row = self.row_by_global_index[idx_i]
            neighbor_slot = self._sample_neighbor_slot(position, row)
            counts[neighbor_slot] += 1
        return counts

    def __getitem__(self, position: int):
        idx_i = self.train_indices[position]
        row = self.row_by_global_index[idx_i]
        neighbor_slot = self._sample_neighbor_slot(position, row)
        idx_j = int(self.neighbor_indices[row, neighbor_slot].item())

        image_i, label_i = self.base_dataset[idx_i]
        image_j, label_j = self.base_dataset[idx_j]

        if self.mode == "class_aware" and int(label_i) != int(label_j):
            raise ValueError(
                f"class_aware pair has mismatched labels for indices {idx_i} and {idx_j}: "
                f"{label_i} != {label_j}"
            )

        if self.anchor_mix_probs is None:
            return image_i, label_i, image_j, label_j, idx_i, idx_j

        mix_prob = float(self.anchor_mix_probs[position].item())
        return image_i, label_i, image_j, label_j, idx_i, idx_j, mix_prob

    def _sample_neighbor_slot(self, position: int, row: int) -> int:
        generator = torch.Generator()
        generator.manual_seed(self._sample_seed(position))

        num_neighbors = int(self.neighbor_indices.shape[1])
        if self.top_rank_only:
            return 0

        if not self.dynamic_neighbor_pool:
            if self.weights is None:
                return int(torch.randint(num_neighbors, size=(1,), generator=generator).item())
            return int(torch.multinomial(self.weights[row], num_samples=1, generator=generator).item())

        pool_size = self._get_neighbor_pool_size(row)
        if pool_size > num_neighbors:
            pool_size = num_neighbors
        if pool_size <= 0:
            pool_size = 1

        if self.weights is None:
            return int(torch.randint(pool_size, size=(1,), generator=generator).item())

        if self.weights.dtype != torch.float32:
            weights = self.weights[row].float()
        else:
            weights = self.weights[row]
        pool_weights = weights[:pool_size]
        pool_weights = pool_weights / pool_weights.sum().clamp_min(1e-12)
        return int(torch.multinomial(pool_weights, num_samples=1, generator=generator).item())

    def _get_neighbor_pool_size(self, row: int) -> int:
        if not self.dynamic_neighbor_pool:
            return int(self.neighbor_indices.shape[1])

        if self.weights is None:
            return self.easy_neighbor_pool_size

        row_weights = self.weights[row]
        top_weight = float(row_weights[0].item()) if row_weights.numel() > 0 else 0.0
        return self.easy_neighbor_pool_size if top_weight >= self.difficulty_threshold else self.hard_neighbor_pool_size

    def _validate_neighbor_pool_settings(self) -> None:
        if self.easy_neighbor_pool_size <= 0:
            raise ValueError("easy_neighbor_pool_size must be positive")
        if self.hard_neighbor_pool_size <= 0:
            raise ValueError("hard_neighbor_pool_size must be positive")

    def _sample_seed(self, position: int) -> int:
        return (self.seed + 1_000_003 * int(self.epoch) + 97_003 * int(position)) % (2**63 - 1)

    def _build_anchor_mix_prob_tensor(
        self,
        anchor_mix_probs: dict[int, float] | None,
    ) -> torch.Tensor | None:
        if anchor_mix_probs is None:
            return None

        missing = [index for index in self.train_indices if index not in anchor_mix_probs]
        if missing:
            raise ValueError(f"Anchor score file is missing train indices: {missing[:10]}")

        values = torch.tensor(
            [float(anchor_mix_probs[index]) for index in self.train_indices],
            dtype=torch.float32,
        )
        if not torch.isfinite(values).all():
            raise ValueError("Anchor mix probabilities must be finite")
        if bool((values < 0).any().item()) or bool((values > 1).any().item()):
            raise ValueError("Anchor mix probabilities must be in [0, 1]")

        return values


def _load_neighbor_index(neighbor_index: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(neighbor_index, (str, Path)):
        payload = torch.load(neighbor_index, map_location="cpu")
    else:
        payload = neighbor_index

    required = {"original_indices", "neighbor_indices"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Neighbor index is missing required keys: {missing}")
    return payload


def _as_long_tensor(value: Any, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    return tensor.cpu()


def _build_index_lookup(original_indices: torch.Tensor) -> dict[int, int]:
    lookup: dict[int, int] = {}
    for row, global_index in enumerate(original_indices.tolist()):
        global_index = int(global_index)
        if global_index in lookup:
            raise ValueError(f"Duplicate original index in neighbor index: {global_index}")
        lookup[global_index] = row
    return lookup
