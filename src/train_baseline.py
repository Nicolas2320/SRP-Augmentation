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
    """Make the experiment more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # More deterministic behavior. This can be slightly slower, but it is useful for research.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_json(path: Path) -> dict:
    """Load one JSON split file created by src/data/make_splits.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find: {path}\n"
            "Run src/data/make_splits.py first."
        )

    with open(path, "r") as f:
        return json.load(f)


def build_train_transform(augmentation: str) -> transforms.Compose:
    """
    Build the training transform.

    none:
        Only converts image to tensor and normalizes it.
        This is preprocessing, not augmentation.

    basic:
        Applies standard CIFAR augmentation:
        - RandomCrop(32, padding=4)
        - RandomHorizontalFlip()
        Then converts to tensor and normalizes.
    """
    if augmentation == "none":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )

    raise ValueError(f"Unknown augmentation: {augmentation}")


def build_eval_transform() -> transforms.Compose:
    """Build validation/test transform. Evaluation must not use random augmentation."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def build_resnet18_cifar10() -> nn.Module:
    """
    Build ResNet18 adapted for CIFAR-10.

    Standard ResNet18 was designed for larger ImageNet images.
    CIFAR-10 images are 32x32, so we use:
    - 3x3 first convolution instead of 7x7
    - stride 1 instead of stride 2
    - no initial maxpool
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


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch and return average loss and accuracy."""
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
    """Evaluate the model without updating weights."""
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
    """Save epoch-by-epoch metrics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "is_best",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ResNet18 baseline on a CIFAR-10 k-shot subset."
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
        "--augmentation",
        type=str,
        default="none",
        choices=["none"],
        help="Augmentation method for training.",
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
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader workers.",
    )

    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="results/metrics",
        help="Directory where metrics will be saved.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="results/checkpoints",
        help="Directory where best model checkpoints will be saved.",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    pin_memory = device.type == "cuda"

    print(f"Using device: {device}")

    split_info = load_json(Path(args.split_path))
    val_split_info = load_json(Path(args.val_split_path))

    train_indices = split_info["train_indices"]
    val_indices = val_split_info["val_indices"]

    dataset_name = split_info.get("dataset", "cifar10")
    k = split_info.get("k", "unknown")
    subset_seed = split_info.get("subset_seed", "unknown")

    if dataset_name != "cifar10":
        raise ValueError(
            f"This script is currently CIFAR-10 only, but split file has dataset={dataset_name}."
        )

    train_transform = build_train_transform(args.augmentation)
    eval_transform = build_eval_transform()

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

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

    run_name = f"cifar10_resnet18_k{k}_seed{subset_seed}_{args.augmentation}"

    metrics_dir = Path(args.metrics_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / f"{run_name}.csv"
    summary_path = metrics_dir / f"{run_name}_summary.json"
    best_model_path = checkpoint_dir / f"{run_name}_best.pt"

    print("Experiment setup:")
    print("  Dataset: CIFAR-10")
    print(f"  Train split file: {args.split_path}")
    print(f"  Validation split file: {args.val_split_path}")
    print(f"  Number of training images: {len(train_dataset)}")
    print(f"  Number of validation images: {len(val_dataset)}")
    print(f"  Number of test images: {len(test_dataset)}")
    print(f"  Augmentation: {args.augmentation}")
    print("  Model: ResNet18 adapted for CIFAR-10")
    print(f"  Epochs: {args.epochs}")
    print(f"  Run name: {run_name}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
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
    best_epoch = 0

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

        is_best = val_acc > best_val_acc

        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "args": vars(args),
                },
                best_model_path,
            )

        metrics.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "is_best": is_best,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        if is_best:
            print(f"  New best model saved: epoch={epoch}, val_acc={val_acc:.4f}")

    if not best_model_path.exists():
        raise RuntimeError("No best model checkpoint was saved. Something went wrong.")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nLoaded best validation model:")
    print(f"  Best epoch: {checkpoint['epoch']}")
    print(f"  Best validation accuracy: {checkpoint['best_val_acc']:.4f}")

    test_loss, test_acc = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\nFinal test result using best validation checkpoint:")
    print(f"  test_loss={test_loss:.4f}")
    print(f"  test_acc={test_acc:.4f}")

    save_metrics_csv(metrics, metrics_path)

    summary = {
        "dataset": "cifar10",
        "model": "resnet18_cifar",
        "augmentation": args.augmentation,
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
        "k": k,
        "subset_seed": subset_seed,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss_best_checkpoint": test_loss,
        "test_acc_best_checkpoint": test_acc,
        "best_model_path": str(best_model_path),
        "metrics_path": str(metrics_path),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved outputs:")
    print(f"  Metrics CSV: {metrics_path}")
    print(f"  Summary JSON: {summary_path}")
    print(f"  Best checkpoint: {best_model_path}")


if __name__ == "__main__":
    main()
