"""
Unified training script for SRP-Augmentation.

Purpose
-------
This file is the official experiment entry point for CIFAR baseline runs.
It replaces the old prototype direction of `train_baseline.py`.

Current v1 support
------------------
- Dataset: CIFAR
- Models: ResNet50, ViT
- Augmentation: none
- k-shot split loading: k5, k10, k20, k50, k100
- subset seed loading: seed0, seed1, seed2 if split files exist
- fixed validation split
- best validation checkpoint saving
- final test evaluation using the best validation checkpoint
- metrics CSV and summary JSON outputs

Future support
--------------
- MixUp
- CutMix
- AugMix

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
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from models.resnet import build_resnet50_cifar
from augmentations.cutmix import CutMix
from augmentations.mixup import apply_mixup, mixup_accuracy, mixup_criterion


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

DatasetName = Literal["cifar10", "cifar100"]
ModelName = Literal["resnet50", "vit"]
AugmentationName = Literal["none", "mixup", "cutmix", "augmix"]


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


# -----------------------------------------------------------------------------
# Reproducibility and device helpers
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



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

    # if model_name == "vit":
    #     Waiting for ViT code

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


def get_transforms(dataset: str, augmentation: str) -> tuple[transforms.Compose, transforms.Compose]:
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

    # These are intentionally not implemented yet.
    # They will be added through the MixUp, CutMix, and AugMix issues.
    if augmentation in {"mixup", "cutmix", "augmix"}:
        train_transform = eval_transform
        return train_transform, eval_transform

    raise ValueError(f"Unsupported augmentation: {augmentation}")



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

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
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
) -> tuple[float, float]:

    model.train()

    total_loss = 0.0
    total_correct = 0.0
    total_examples = 0

    for images, targets in dataloader:

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        # =====================================================
        # CUTMIX
        # =====================================================

        if augmentation == "cutmix":

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

        # =====================================================
        # MIXUP
        # =====================================================

        elif augmentation == "mixup":

            images, targets_a, targets_b, lam = apply_mixup(
                images=images,
                targets=targets,
                alpha=mixup_alpha,
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

        # =====================================================
        # NO AUGMENTATION
        # =====================================================

        else:

            logits = model(images)

            loss = criterion(logits, targets)

            batch_correct = (logits.argmax(dim=1) == targets).sum().item()

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)

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
    return (
        f"{config.dataset}_{config.model}_"
        f"k{config.k}_seed{config.subset_seed}_"
        f"{config.augmentation}_epochs{config.epochs}"
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
    
    cutmix = None

    if config.augmentation == "cutmix":
        cutmix = CutMix(alpha=1.0)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    metrics: list[dict] = []
    best_val_acc = -math.inf
    best_epoch = -1
    

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            augmentation=config.augmentation,
            cutmix=cutmix,
            mixup_alpha=config.mixup_alpha,
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
            f"val_acc={val_acc:.4f}{marker}"
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
    choices=['none', 'mixup', 'cutmix'],
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
    )


if __name__ == "__main__":
    experiment_config = parse_args()
    run_experiment(experiment_config)
