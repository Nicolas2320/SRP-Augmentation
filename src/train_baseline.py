import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed: int) -> None:
    """
    Makes the experiment more reproducible.

    This controls Python randomness, NumPy randomness, and PyTorch randomness.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Selects the best available device.

    Priority:
    1. CUDA GPU
    2. Apple MPS GPU
    3. CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_json(path: Path) -> dict:
    """
    Loads one of the split files created by make_splits.py.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find: {path}\n"
            "Run src/data/make_splits.py first."
        )

    with open(path, "r") as f:
        return json.load(f)


def build_resnet18_cifar10() -> nn.Module:
    """
    Builds a ResNet18 adapted for CIFAR-10.

    Standard ResNet18 was originally designed for larger ImageNet images.
    CIFAR-10 images are only 32x32, so we use:
    - smaller first convolution: 3x3 instead of 7x7
    - stride 1 instead of stride 2
    - remove the initial maxpool

    This is a common adaptation for CIFAR-style experiments.
    """
    model = resnet18(weights=None, num_classes=10)

    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    model.maxpool = nn.Identity()

    return model


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes classification accuracy for one batch.
    """
    predictions = logits.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return correct / total


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Runs one training epoch.

    Returns:
    - average training loss
    - average training accuracy
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples

    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluates the model without updating weights.

    Returns:
    - average loss
    - average accuracy
    """
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

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples

    return avg_loss, avg_acc


def save_metrics_csv(metrics: list[dict], output_path: Path) -> None:
    """
    Saves epoch-by-epoch metrics to a CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a first ResNet18 baseline on CIFAR-10 k-shot subset."
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="data/raw",
        help="Path where CIFAR-10 is stored/downloaded.",
    )

    parser.add_argument(
        "--split-path",
        type=str,
        default="data/splits/cifar10/k20_seed0.json",
        help="Path to the k-shot training subset JSON file.",
    )

    parser.add_argument(
        "--val-split-path",
        type=str,
        default="data/splits/cifar10/fixed_validation_split.json",
        help="Path to the fixed validation split JSON file.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Training seed.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/metrics",
        help="Directory where metrics will be saved.",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    print(f"Using device: {device}")

    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_full = CIFAR10(
        root=args.data_root,
        train=True,
        download=True,
        transform=train_transform,
    )

    val_full = CIFAR10(
        root=args.data_root,
        train=True,
        download=True,
        transform=eval_transform,
    )

    test_dataset = CIFAR10(
        root=args.data_root,
        train=False,
        download=True,
        transform=eval_transform,
    )

    split_info = load_json(Path(args.split_path))
    val_split_info = load_json(Path(args.val_split_path))

    train_indices = split_info["train_indices"]
    val_indices = val_split_info["val_indices"]

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

    print("Experiment setup:")
    print(f"  Dataset: CIFAR-10")
    print(f"  Train split file: {args.split_path}")
    print(f"  Validation split file: {args.val_split_path}")
    print(f"  Number of training images: {len(train_dataset)}")
    print(f"  Number of validation images: {len(val_dataset)}")
    print(f"  Number of test images: {len(test_dataset)}")
    print(f"  Augmentation: none")
    print(f"  Model: ResNet18 adapted for CIFAR-10")
    print(f"  Epochs: {args.epochs}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = build_resnet18_cifar10().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    metrics = []

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        metrics.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

    test_loss, test_acc = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\nFinal test result:")
    print(f"  test_loss={test_loss:.4f}")
    print(f"  test_acc={test_acc:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "cifar10_resnet18_k20_seed0_noaug.csv"
    save_metrics_csv(metrics, metrics_path)

    summary_path = output_dir / "cifar10_resnet18_k20_seed0_noaug_summary.json"

    summary = {
        "dataset": "cifar10",
        "model": "resnet18_cifar",
        "augmentation": "none",
        "split_path": args.split_path,
        "val_split_path": args.val_split_path,
        "num_train": len(train_dataset),
        "num_val": len(val_dataset),
        "num_test": len(test_dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved outputs:")
    print(f"  Metrics CSV: {metrics_path}")
    print(f"  Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()