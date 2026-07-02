"""
Unified training script for SRP-Augmentation.

Purpose
-------
This file is the official experiment entry point for CIFAR baseline runs.
It replaces the old prototype direction of `train_baseline.py`.

Current v1 support
------------------
- Dataset: CIFAR-10, CIFAR-100
- Models: ResNet50, ViT
- Augmentation: none, MixUp, CutMix, AugMix
- k-shot split loading: k5, k10, k20, k50, k100
- subset seed loading: seed0, seed1, seed2 if split files exist
- fixed validation split
- best validation checkpoint saving
- final test evaluation using the best validation checkpoint
- metrics CSV and summary JSON outputs

Example
-------
python src/train.py \
  --dataset cifar100 \
  --model resnet50 \
  --k 20 \
  --subset-seed 0 \
  --augmentation none \
  --epochs 10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, get_worker_info
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from models.resnet import build_resnet50_cifar
from models.vit import build_vit_cifar
from augmentations.cutmix import CutMix
from augmentations.augmix import AugMixTransform
from augmentations.mixup import apply_mixup, mixup_accuracy, mixup_criterion
from augmentations.similarity_guided import (
    apply_simcutmix,
    apply_simmixup,
    simmixup_accuracy,
    simmixup_criterion,
)
from data.indexed_dataset import GuidedPairDataset


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

DatasetName = Literal["cifar10", "cifar100"]
ModelName = Literal["resnet50", "vit"]
AugmentationName = Literal["none", "mixup", "cutmix", "augmix", "simmixup", "simcutmix"]


@dataclass
class ExperimentConfig:
    dataset: str
    model: str
    k: int
    subset_seed: int
    augmentation: str
    mixup_alpha: float
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    train_seed: int
    data_root: str
    split_root: str
    output_root: str
    num_workers: int
    neighbor_path: str | None
    guided_mode: str
    neighbor_k: int
    neighbor_rank_start: int
    pair_sampling: str
    mix_prob: float
    mix_warmup_epochs: int
    anchor_score_path: str | None
    anchor_selection: str
    anchor_top_pct: float
    anchor_score_power: float


# -----------------------------------------------------------------------------
# Reproducibility and device helpers
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set random seeds and deterministic CUDA/cuDNN behavior where possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader worker RNGs and any transform-local RNG."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

    worker_info = get_worker_info()
    if worker_info is None:
        return

    dataset = worker_info.dataset
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    transform = getattr(base_dataset, "transform", None)
    if hasattr(transform, "set_seed"):
        transform.set_seed(worker_seed)



def get_device() -> torch.device:
    """Select CUDA, Apple MPS, or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Model builders
# -----------------------------------------------------------------------------


# ResNet builder is implemented in `src/models/resnet.py` and imported above.

def build_model(model_name: str, num_classes: int = 100) -> nn.Module:
    """Build selected model."""
    if model_name == "resnet50":
        return build_resnet50_cifar(num_classes=num_classes)

    if model_name == "vit":
        return build_vit_cifar(num_classes=num_classes)

    raise ValueError(f"Unsupported model: {model_name}")

# -----------------------------------------------------------------------------
# Data and transforms
# -----------------------------------------------------------------------------


def load_json(path: Path) -> dict:
    """Load a JSON file and give a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}.\n"
            "Make sure you already ran `python src/data/make_splits.py`."
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_num_classes(dataset: str) -> int:
    if dataset == "cifar10":
        return 10
    if dataset == "cifar100":
        return 100
    raise ValueError(f"Unsupported dataset: {dataset}")


def get_dataset_class(dataset: str):
    if dataset == "cifar10":
        return CIFAR10
    if dataset == "cifar100":
        return CIFAR100
    raise ValueError(f"Unsupported dataset: {dataset}")


def get_dataset_stats(dataset: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if dataset == "cifar10":
        return CIFAR10_MEAN, CIFAR10_STD
    if dataset == "cifar100":
        return CIFAR100_MEAN, CIFAR100_STD
    raise ValueError(f"Unsupported dataset: {dataset}")

def get_split_paths(dataset: str, k: int, subset_seed: int, split_root: str) -> tuple[Path, Path]:
    """Return train subset and validation split paths."""
    if dataset not in {"cifar10", "cifar100"}:
        raise ValueError(f"Unsupported dataset: {dataset}")

    split_dir = Path(split_root) / dataset
    train_split_path = split_dir / f"k{k}_seed{subset_seed}.json"
    val_split_path = split_dir / "fixed_validation_split.json"

    return train_split_path, val_split_path


def get_transforms(
    dataset: str,
    augmentation: str,
    seed: int,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Return train and evaluation transforms."""
    mean, std = get_dataset_stats(dataset)

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    if augmentation == "none":
        train_transform = eval_transform
        return train_transform, eval_transform

    if augmentation == "augmix":
        train_transform = AugMixTransform(
            mean=mean,
            std=std,
            severity=3,
            width=3,
            depth=-1,
            alpha=1.0,
            seed=seed,
        )
        return train_transform, eval_transform

    if augmentation in {"mixup", "cutmix", "simmixup", "simcutmix"}:
        train_transform = eval_transform
        return train_transform, eval_transform

    raise ValueError(f"Unsupported augmentation: {augmentation}")


def load_neighbor_payload(
    path: str,
    guided_mode: str,
    neighbor_k: int,
    neighbor_rank_start: int = 1,
) -> dict[str, Any]:
    """Load and optionally truncate a saved neighbor payload."""
    neighbor_path = Path(path)
    if not neighbor_path.exists():
        raise FileNotFoundError(f"Missing neighbor file: {neighbor_path}")
    if neighbor_k < 1:
        raise ValueError("--neighbor-k must be at least 1")
    if neighbor_rank_start < 1:
        raise ValueError("--neighbor-rank-start must be at least 1")

    payload = torch.load(neighbor_path, map_location="cpu")
    required = {"mode", "original_indices", "neighbor_indices"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Neighbor file is missing required keys: {missing}")

    if payload["mode"] != guided_mode:
        raise ValueError(
            f"Neighbor file mode is {payload['mode']!r}, but --guided-mode is {guided_mode!r}"
        )

    neighbor_indices = torch.as_tensor(payload["neighbor_indices"], dtype=torch.long)
    if neighbor_indices.ndim != 2:
        raise ValueError("neighbor_indices must be a 2D tensor")
    rank_start_idx = neighbor_rank_start - 1
    rank_end_idx = rank_start_idx + neighbor_k
    rank_end = neighbor_rank_start + neighbor_k - 1
    if neighbor_indices.shape[1] < rank_end_idx:
        raise ValueError(
            f"Requested neighbor rank window {neighbor_rank_start}-{rank_end} "
            f"exceeds saved neighbor count {neighbor_indices.shape[1]}"
        )

    payload = dict(payload)
    payload["neighbor_indices"] = neighbor_indices[:, rank_start_idx:rank_end_idx].clone()
    if "similarities" in payload:
        payload["similarities"] = torch.as_tensor(payload["similarities"], dtype=torch.float32)[
            :, rank_start_idx:rank_end_idx
        ].clone()
    payload["num_neighbors"] = int(neighbor_k)
    payload["neighbor_rank_start"] = int(neighbor_rank_start)
    payload["neighbor_rank_end"] = int(rank_end)
    return payload


def load_anchor_mix_probabilities(
    path: str | None,
    train_indices: list[int] | tuple[int, ...],
    selection: str,
    top_pct: float,
    score_power: float,
    seed: int,
) -> dict[int, float] | None:
    """Load anchor scores and convert them into per-anchor mix probabilities."""
    if path is None:
        return None

    if selection not in {"top_fraction", "score_probability", "random_fraction"}:
        raise ValueError(f"Unsupported anchor selection: {selection}")
    if not 0.0 <= top_pct <= 1.0:
        raise ValueError("--anchor-top-pct must be in [0, 1]")
    if score_power <= 0:
        raise ValueError("--anchor-score-power must be positive")

    score_path = Path(path)
    if not score_path.exists():
        raise FileNotFoundError(f"Missing anchor score file: {score_path}")

    payload = torch.load(score_path, map_location="cpu")
    required = {"original_indices", "score"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Anchor score file is missing required keys: {missing}")

    original_indices = torch.as_tensor(payload["original_indices"], dtype=torch.long)
    scores = torch.as_tensor(payload["score"], dtype=torch.float32)
    if original_indices.ndim != 1 or scores.ndim != 1:
        raise ValueError("original_indices and score must be 1D tensors")
    if original_indices.numel() != scores.numel():
        raise ValueError("original_indices and score must have matching lengths")
    if not torch.isfinite(scores).all():
        raise ValueError("Anchor scores must be finite")

    row_by_index = {int(index): row for row, index in enumerate(original_indices.tolist())}
    missing_indices = [int(index) for index in train_indices if int(index) not in row_by_index]
    if missing_indices:
        raise ValueError(f"Anchor score file is missing train indices: {missing_indices[:10]}")

    aligned_scores = torch.tensor(
        [float(scores[row_by_index[int(index)]].item()) for index in train_indices],
        dtype=torch.float32,
    )
    mix_probs = torch.zeros(len(train_indices), dtype=torch.float32)

    if selection == "score_probability":
        score_percentiles = rank_percentile(aligned_scores)
        mix_probs = torch.clamp(score_percentiles.pow(float(score_power)), min=0.0, max=1.0)
    else:
        selected_count = int(math.ceil(len(train_indices) * top_pct))
        if selected_count > 0:
            if selection == "top_fraction":
                selected_positions = torch.argsort(aligned_scores, descending=True)[:selected_count]
            else:
                generator = torch.Generator()
                generator.manual_seed(int(seed))
                selected_positions = torch.randperm(len(train_indices), generator=generator)[:selected_count]
            mix_probs[selected_positions] = 1.0

    return {
        int(index): float(prob)
        for index, prob in zip(train_indices, mix_probs.tolist())
    }


def rank_percentile(values: torch.Tensor) -> torch.Tensor:
    """Return simple rank percentiles in [0, 1] for a 1D tensor."""
    if values.ndim != 1:
        raise ValueError(f"rank_percentile expects a 1D tensor, got {tuple(values.shape)}")
    if values.numel() == 1:
        return torch.ones_like(values, dtype=torch.float32)

    order = torch.argsort(values)
    ranks = torch.empty(values.numel(), dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float32)
    return ranks / float(values.numel() - 1)



def build_dataloaders(
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader, int, int, int]:
    """Build train, validation, and test dataloaders."""
    train_split_path, val_split_path = get_split_paths(
        dataset=config.dataset,
        k=config.k,
        subset_seed=config.subset_seed,
        split_root=config.split_root,
    )

    train_split_info = load_json(train_split_path)
    val_split_info = load_json(val_split_path)

    train_indices = train_split_info["train_indices"]
    val_indices = val_split_info["val_indices"]

    train_transform, eval_transform = get_transforms(
        dataset=config.dataset,
        augmentation=config.augmentation,
        seed=config.train_seed,
    )

    DatasetClass = get_dataset_class(config.dataset)

    train_full = DatasetClass(
        root=config.data_root,
        train=True,
        download=True, # False on cluster
        transform=train_transform,
    )

    val_full = DatasetClass(
        root=config.data_root,
        train=True,
        download=True, # False on cluster
        transform=eval_transform,
    )

    test_dataset = DatasetClass(
        root=config.data_root,
        train=False,
        download=True, # False on cluster
        transform=eval_transform,
    )

    if config.augmentation in {"simmixup", "simcutmix"}:
        if config.neighbor_path is None:
            raise ValueError(f"--neighbor-path is required when --augmentation {config.augmentation}")
        neighbor_payload = load_neighbor_payload(
            path=config.neighbor_path,
            guided_mode=config.guided_mode,
            neighbor_k=config.neighbor_k,
            neighbor_rank_start=config.neighbor_rank_start,
        )
        anchor_mix_probs = load_anchor_mix_probabilities(
            path=config.anchor_score_path,
            train_indices=train_indices,
            selection=config.anchor_selection,
            top_pct=config.anchor_top_pct,
            score_power=config.anchor_score_power,
            seed=config.train_seed,
        )
        train_dataset = GuidedPairDataset(
            base_dataset=train_full,
            train_indices=train_indices,
            neighbor_index=neighbor_payload,
            pair_sampling=config.pair_sampling,
            mode=config.guided_mode,
            seed=config.train_seed,
            anchor_mix_probs=anchor_mix_probs,
        )
    else:
        train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

    pin_memory = device.type == "cuda"
    train_generator = torch.Generator()
    train_generator.manual_seed(config.train_seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(config.train_seed + 1)
    test_generator = torch.Generator()
    test_generator.manual_seed(config.train_seed + 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=val_generator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=test_generator,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    augmentation: str = "none",
    cutmix=None,
    mixup_alpha: float = 1.0,
    mix_prob: float = 1.0,
    augmentation_rng: np.random.Generator | None = None,
) -> tuple[float, float]:

    model.train()

    total_loss = 0.0
    total_correct = 0.0
    total_examples = 0

    for batch in dataloader:

        optimizer.zero_grad(set_to_none=True)

        # =====================================================
        # SIMCUTMIX
        # =====================================================

        if augmentation == "simcutmix":

            images_i, targets_i, images_j, targets_j, *extra = batch
            images_i = images_i.to(device)
            targets_i = targets_i.to(device)
            images_j = images_j.to(device)
            targets_j = targets_j.to(device)
            anchor_mix_prob = extra[2].to(device) if len(extra) >= 3 else None

            images, targets_a, targets_b, lam = apply_simcutmix(
                images_i=images_i,
                targets_i=targets_i,
                images_j=images_j,
                targets_j=targets_j,
                alpha=mixup_alpha,
                mix_prob=mix_prob,
                sample_mix_prob=anchor_mix_prob,
                rng=augmentation_rng,
            )

            logits = model(images)

            loss = simmixup_criterion(
                criterion=criterion,
                predictions=logits,
                targets_i=targets_a,
                targets_j=targets_b,
                lam=lam,
            )

            batch_correct = simmixup_accuracy(
                predictions=logits,
                targets_i=targets_a,
                targets_j=targets_b,
                lam=lam,
            )

            batch_size = targets_i.size(0)

        # =====================================================
        # SIMMIXUP
        # =====================================================

        elif augmentation == "simmixup":

            images_i, targets_i, images_j, targets_j, *extra = batch
            images_i = images_i.to(device)
            targets_i = targets_i.to(device)
            images_j = images_j.to(device)
            targets_j = targets_j.to(device)
            anchor_mix_prob = extra[2].to(device) if len(extra) >= 3 else None

            images, targets_a, targets_b, lam = apply_simmixup(
                images_i=images_i,
                targets_i=targets_i,
                images_j=images_j,
                targets_j=targets_j,
                alpha=mixup_alpha,
                mix_prob=mix_prob,
                sample_mix_prob=anchor_mix_prob,
                rng=augmentation_rng,
            )

            logits = model(images)

            loss = simmixup_criterion(
                criterion=criterion,
                predictions=logits,
                targets_i=targets_a,
                targets_j=targets_b,
                lam=lam,
            )

            batch_correct = simmixup_accuracy(
                predictions=logits,
                targets_i=targets_a,
                targets_j=targets_b,
                lam=lam,
            )

            batch_size = targets_i.size(0)

        # =====================================================
        # CUTMIX
        # =====================================================

        elif augmentation == "cutmix":

            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)

            images, targets_a, targets_b, lam = cutmix(images, targets)

            logits = model(images)

            loss = (
                lam * criterion(logits, targets_a)
                + (1.0 - lam) * criterion(logits, targets_b)
            )

            predicted = logits.argmax(dim=1)
            batch_correct = (
                lam * predicted.eq(targets_a).float()
                + (1.0 - lam) * predicted.eq(targets_b).float()
            ).sum().item()

            batch_size = targets.size(0)

        # =====================================================
        # MIXUP
        # =====================================================

        elif augmentation == "mixup":

            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)

            images, targets_a, targets_b, lam = apply_mixup(
                images=images,
                targets=targets,
                alpha=mixup_alpha,
                rng=augmentation_rng,
            )

            logits = model(images)

            loss = mixup_criterion(
                criterion=criterion,
                predictions=logits,
                targets_a=targets_a,
                targets_b=targets_b,
                lam=lam,
            )

            batch_correct = mixup_accuracy(
                predictions=logits,
                targets_a=targets_a,
                targets_b=targets_b,
                lam=lam,
            )

            batch_size = targets.size(0)

        # =====================================================
        # NO AUGMENTATION
        # =====================================================

        else:

            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)

            loss = criterion(logits, targets)

            batch_correct = (logits.argmax(dim=1) == targets).sum().item()

            batch_size = targets.size(0)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_correct += batch_correct
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model and return average loss and accuracy."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples

# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def experiment_name(config: ExperimentConfig) -> str:
    """Create a consistent experiment name for files."""
    base_name = (
        f"{config.dataset}_{config.model}_"
        f"k{config.k}_seed{config.subset_seed}_"
        f"{config.augmentation}"
    )

    if config.augmentation not in {"simmixup", "simcutmix"}:
        return f"{base_name}_epochs{config.epochs}"

    alpha = format_float_for_filename(config.mixup_alpha)
    mix_prob = format_float_for_filename(config.mix_prob)
    rank_start = int(config.neighbor_rank_start)
    rank_end = rank_start + int(config.neighbor_k) - 1
    neighbor_source = filename_component(
        Path(config.neighbor_path).stem if config.neighbor_path else "no_neighbor_file"
    )
    anchor_part = ""
    if config.anchor_score_path is not None:
        anchor_source = filename_component(Path(config.anchor_score_path).stem)
        top_pct = format_float_for_filename(config.anchor_top_pct)
        score_power = format_float_for_filename(config.anchor_score_power)
        anchor_part = (
            f"_anchor_{anchor_source}_{config.anchor_selection}_"
            f"top{top_pct}_pow{score_power}"
        )
    return (
        f"{base_name}_{config.guided_mode}_"
        f"{neighbor_source}_"
        f"nk{config.neighbor_k}_r{rank_start}-{rank_end}_{config.pair_sampling}_"
        f"alpha{alpha}_mp{mix_prob}_warm{config.mix_warmup_epochs}_"
        f"epochs{config.epochs}{anchor_part}"
    )


def format_float_for_filename(value: float) -> str:
    """Format a float compactly for deterministic filename components."""
    return filename_component(f"{value:g}")


def filename_component(value: str) -> str:
    """Make a short value safe for use inside experiment filenames."""
    safe_chars = []
    for character in str(value):
        if character.isalnum() or character in {"_", "-"}:
            safe_chars.append(character)
        elif character == ".":
            safe_chars.append("p")
        else:
            safe_chars.append("_")
    return "".join(safe_chars)


def format_neighbor_rank_summary(train_dataset: Any, effective_mix_prob: float) -> str:
    """Format sampled SimMixUp neighbor-rank counts for epoch logging."""

    rank_counter = getattr(train_dataset, "sampled_neighbor_rank_counts", None)
    if not callable(rank_counter):
        return ""

    neighbor_k = int(getattr(train_dataset, "neighbor_indices").shape[1])
    rank_start = int(getattr(train_dataset, "neighbor_rank_start", 1))
    rank_end = rank_start + neighbor_k - 1
    if effective_mix_prob <= 0.0:
        return (
            f" | neighbor_k={neighbor_k} | "
            f"neighbor_rank_window={rank_start}-{rank_end} disabled"
        )

    counts = rank_counter()
    total = int(counts.sum().item())
    if total == 0:
        return (
            f" | neighbor_k={neighbor_k} | "
            f"neighbor_rank_window={rank_start}-{rank_end} none"
        )

    ranks = torch.arange(rank_start, rank_end + 1, dtype=torch.float32)
    mean_rank = float((counts.float() * ranks).sum().item() / total)
    nonzero = torch.nonzero(counts, as_tuple=False).flatten()
    min_rank = int(nonzero[0].item()) + rank_start
    max_rank = int(nonzero[-1].item()) + rank_start
    count_text = ",".join(
        f"{rank}:{int(count)}"
        for rank, count in enumerate(counts.tolist(), start=rank_start)
    )

    return (
        f" | neighbor_k={neighbor_k} | "
        f"neighbor_rank_window={rank_start}-{rank_end} | "
        f"sampled_rank={min_rank}-{max_rank} mean={mean_rank:.2f} | "
        f"rank_counts={count_text}"
    )


def format_anchor_mix_summary(train_dataset: Any, effective_mix_prob: float) -> str:
    """Format score-gated anchor mix coverage for epoch logging."""
    anchor_probs = getattr(train_dataset, "anchor_mix_probs", None)
    if anchor_probs is None:
        return ""

    effective_probs = anchor_probs.float() * float(effective_mix_prob)
    active_count = int((effective_probs > 0).sum().item())
    mean_prob = float(effective_probs.mean().item())
    max_prob = float(effective_probs.max().item()) if effective_probs.numel() else 0.0
    return (
        f" | anchor_mix={active_count}/{effective_probs.numel()} "
        f"mean_prob={mean_prob:.3f} max_prob={max_prob:.3f}"
    )


def save_metrics_csv(metrics: list[dict], output_path: Path) -> None:
    """Save epoch-level metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)



def save_json(path: Path, data: dict) -> None:
    """Save dictionary as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# -----------------------------------------------------------------------------
# Main experiment logic
# -----------------------------------------------------------------------------


def run_experiment(config: ExperimentConfig) -> None:
    """Run one training experiment."""
    set_seed(config.train_seed)
    device = get_device()

    name = experiment_name(config)
    output_root = Path(config.output_root)

    metrics_path = output_root / "metrics" / f"{name}.csv"
    summary_path = output_root / "metrics" / f"{name}_summary.json"
    checkpoint_path = output_root / "checkpoints" / f"{name}_best.pt"

    print("Experiment configuration:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")
    print(f"  device: {device}")
    print(f"  experiment_name: {name}")

    (
        train_loader,
        val_loader,
        test_loader,
        num_train,
        num_val,
        num_test,
    ) = build_dataloaders(config=config, device=device)

    num_classes = get_num_classes(config.dataset)
    model = build_model(config.model, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    augmentation_rng = np.random.default_rng(config.train_seed)
    
    cutmix = None

    if config.augmentation == "cutmix":
        cutmix = CutMix(alpha=1.0, seed=config.train_seed)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    metrics: list[dict] = []
    best_val_acc = -math.inf
    best_epoch = -1
    

    for epoch in range(1, config.epochs + 1):
        train_dataset = getattr(train_loader, "dataset", None)
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)

        effective_mix_prob = config.mix_prob
        if config.augmentation in {"simmixup", "simcutmix"} and epoch <= config.mix_warmup_epochs:
            effective_mix_prob = 0.0
        neighbor_rank_summary = format_neighbor_rank_summary(
            train_dataset=train_dataset,
            effective_mix_prob=effective_mix_prob,
        )
        anchor_mix_summary = format_anchor_mix_summary(
            train_dataset=train_dataset,
            effective_mix_prob=effective_mix_prob,
        )

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            augmentation=config.augmentation,
            cutmix=cutmix,
            mixup_alpha=config.mixup_alpha,
            mix_prob=effective_mix_prob,
            augmentation_rng=augmentation_rng,
        )

        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        metrics.append(row)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "config": asdict(config),
                },
                checkpoint_path,
            )

        marker = " * Checkpoint - Best val_acc " if improved else ""
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
            f"{neighbor_rank_summary}{anchor_mix_summary}{marker}"
        )

    save_metrics_csv(metrics, metrics_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    summary = {
        **asdict(config),
        "num_train": num_train,
        "num_val": num_val,
        "num_test": num_test,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss_best_checkpoint": test_loss,
        "test_acc_best_checkpoint": test_acc,
        "metrics_path": str(metrics_path),
        "best_model_path": str(checkpoint_path),
    }

    save_json(summary_path, summary)

    print("\nFinal result using best validation checkpoint:")
    print(f"  best_epoch: {best_epoch}")
    print(f"  best_val_acc: {best_val_acc:.4f}")
    print(f"  test_loss: {test_loss:.4f}")
    print(f"  test_acc: {test_acc:.4f}")
    print("\nSaved outputs:")
    print(f"  metrics: {metrics_path}")
    print(f"  summary: {summary_path}")
    print(f"  checkpoint: {checkpoint_path}")



def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Unified CIFAR SRP training script")

    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100"])
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "vit"])
    parser.add_argument("--k", type=int, default=20, choices=[5, 10, 20, 50, 100])
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument(
    '--augmentation',
    type=str,
    default='none',
    choices=["none", "mixup", "cutmix", "augmix", "simmixup", "simcutmix"],
    help='augmentation method'
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--split-root", type=str, default="data/splits")
    parser.add_argument("--output-root", type=str, default="results")
    parser.add_argument( # this is just for mixup
        "--mixup-alpha",
        type=float,
        default=1.0,
        help="MixUp alpha parameter. Default: 1.0.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Use 0 on Windows for fewer DataLoader issues.",
    )
    parser.add_argument(
        "--neighbor-path",
        type=str,
        default=None,
        help="Path to saved neighbor payload for similarity-guided augmentation.",
    )
    parser.add_argument(
        "--guided-mode",
        type=str,
        default="class_aware",
        choices=["class_aware", "class_agnostic", "different_label"],
        help="Neighbor mode expected in --neighbor-path.",
    )
    parser.add_argument(
        "--neighbor-k",
        type=int,
        default=10,
        help="Number of nearest neighbors to sample from.",
    )
    parser.add_argument(
        "--neighbor-rank-start",
        type=int,
        default=1,
        help=(
            "1-indexed first neighbor rank to sample from. Combined with "
            "--neighbor-k to define the inclusive rank window."
        ),
    )
    parser.add_argument(
        "--pair-sampling",
        type=str,
        default="uniform",
        choices=["uniform", "weighted"],
        help="How to sample partners from each saved neighbor set.",
    )
    parser.add_argument(
        "--mix-prob",
        type=float,
        default=1.0,
        help="Probability of applying SimMixUp to a paired batch.",
    )
    parser.add_argument(
        "--mix-warmup-epochs",
        type=int,
        default=0,
        help="Disable SimMixUp for the first N epochs.",
    )
    parser.add_argument(
        "--anchor-score-path",
        type=str,
        default=None,
        help="Optional score payload for targeted SimMixUp/SimCutMix anchors.",
    )
    parser.add_argument(
        "--anchor-selection",
        type=str,
        default="top_fraction",
        choices=["top_fraction", "score_probability", "random_fraction"],
        help="How to convert anchor scores into per-anchor mix probabilities.",
    )
    parser.add_argument(
        "--anchor-top-pct",
        type=float,
        default=0.2,
        help="Fraction of anchors selected when using top_fraction or random_fraction.",
    )
    parser.add_argument(
        "--anchor-score-power",
        type=float,
        default=1.0,
        help="Exponent for score_probability anchor mixing.",
    )

    args = parser.parse_args()

    return ExperimentConfig(
        dataset=args.dataset,
        model=args.model,
        k=args.k,
        subset_seed=args.subset_seed,
        augmentation=args.augmentation,
        mixup_alpha=args.mixup_alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_seed=args.train_seed,
        data_root=args.data_root,
        split_root=args.split_root,
        output_root=args.output_root,
        num_workers=args.num_workers,
        neighbor_path=args.neighbor_path,
        guided_mode=args.guided_mode,
        neighbor_k=args.neighbor_k,
        neighbor_rank_start=args.neighbor_rank_start,
        pair_sampling=args.pair_sampling,
        mix_prob=args.mix_prob,
        mix_warmup_epochs=args.mix_warmup_epochs,
        anchor_score_path=args.anchor_score_path,
        anchor_selection=args.anchor_selection,
        anchor_top_pct=args.anchor_top_pct,
        anchor_score_power=args.anchor_score_power,
    )


if __name__ == "__main__":
    experiment_config = parse_args()
    run_experiment(experiment_config)
