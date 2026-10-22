"""Extract, verify, and cache prerequisites for Tiny ImageNet experiments."""

from __future__ import annotations

import argparse
import zipfile
from collections import Counter
from pathlib import Path

from torchvision.models import ResNet50_Weights, resnet50

try:
    from .tiny_imagenet import TINY_IMAGENET_DIRNAME, TinyImageNet
except ImportError:  # Support `python src/data/prepare_tiny_imagenet.py`.
    from tiny_imagenet import TINY_IMAGENET_DIRNAME, TinyImageNet


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract a zip archive after rejecting absolute and traversal paths."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        handle.extractall(destination)


def verify_dataset(data_root: Path) -> tuple[int, int]:
    train = TinyImageNet(data_root, split="train")
    labeled_test = TinyImageNet(data_root, split="val")
    if len(train.classes) != 200:
        raise ValueError(f"Expected 200 Tiny ImageNet classes; found {len(train.classes)}")
    train_counts = Counter(train.targets)
    test_counts = Counter(labeled_test.targets)
    if set(train_counts.values()) != {500}:
        raise ValueError("Expected exactly 500 official training images per class")
    if set(test_counts.values()) != {50}:
        raise ValueError("Expected exactly 50 labeled official-validation images per class")
    if len(train) != 100_000 or len(labeled_test) != 10_000:
        raise ValueError("Unexpected Tiny ImageNet partition sizes")
    return len(train), len(labeled_test)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--archive",
        type=Path,
        help="Optional local tiny-imagenet-200.zip to extract when the dataset is absent.",
    )
    parser.add_argument(
        "--cache-resnet50",
        action="store_true",
        help="Load ImageNet ResNet50 weights now so compute nodes need no download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.data_root / TINY_IMAGENET_DIRNAME
    if not (dataset_root / "wnids.txt").is_file():
        if args.archive is None:
            raise FileNotFoundError(
                f"Missing {dataset_root}. Supply --archive tiny-imagenet-200.zip."
            )
        if not args.archive.is_file():
            raise FileNotFoundError(f"Tiny ImageNet archive not found: {args.archive}")
        args.data_root.mkdir(parents=True, exist_ok=True)
        safe_extract(args.archive, args.data_root)

    train_count, test_count = verify_dataset(args.data_root)
    print(f"Tiny ImageNet verified: train={train_count}, labeled_test={test_count}")
    print("Policy: reserve 50/class from official train; official labeled val is final test")
    print("Policy: official unlabeled test partition is unused")

    if args.cache_resnet50:
        resnet50(weights=ResNet50_Weights.DEFAULT)
        checkpoint = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        print(f"ImageNet ResNet50 weights cached below: {checkpoint}")


if __name__ == "__main__":
    main()
