"""Compute feature embeddings for a k-shot CIFAR training subset only.

This script intentionally reads only the k-shot training split JSON. It checks
that none of those indices overlap with the fixed validation split, and it uses
the torchvision training partition (`train=True`) rather than validation or test
images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import ResNet50_Weights, resnet50


DATASET_CLASSES = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
}


class IndexedSubsetDataset(Dataset):
    """Dataset wrapper that returns original CIFAR train indices."""

    def __init__(self, base_dataset: Dataset, indices: list[int]) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int, int]:
        original_index = int(self.indices[position])
        image, label = self.base_dataset[original_index]
        return image, int(label), original_index


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_ints(values: list[int]) -> str:
    payload = json.dumps([int(v) for v in values], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def output_dir(output_root: Path, dataset: str, k: int, subset_seed: int) -> Path:
    return output_root / dataset / f"k{k}_seed{subset_seed}"


def build_encoder(name: str, device: torch.device) -> tuple[nn.Module, Any, int]:
    if name != "resnet50_imagenet":
        raise ValueError(f"Unsupported encoder: {name}")

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    feature_extractor.to(device)
    return feature_extractor, weights.transforms(), 2048


def get_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def compute_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    original_indices: list[torch.Tensor] = []

    for images, batch_labels, batch_indices in dataloader:
        images = images.to(device)
        batch_embeddings = model(images).flatten(start_dim=1).cpu()
        embeddings.append(batch_embeddings)
        labels.append(batch_labels.cpu().long())
        original_indices.append(batch_indices.cpu().long())

    return (
        torch.cat(embeddings, dim=0),
        torch.cat(labels, dim=0),
        torch.cat(original_indices, dim=0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute encoder embeddings for the k-shot training subset only."
    )
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--subset-seed", type=int, required=True)
    parser.add_argument("--encoder", choices=["resnet50_imagenet"], required=True)
    parser.add_argument("--split-root", type=str, default="data/splits")
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--output-root", type=str, default="results/neighbors")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_root = Path(args.split_root)
    data_root = Path(args.data_root)
    out_dir = output_dir(Path(args.output_root), args.dataset, args.k, args.subset_seed)

    train_split_path = split_root / args.dataset / f"k{args.k}_seed{args.subset_seed}.json"
    val_split_path = split_root / args.dataset / "fixed_validation_split.json"
    train_split = load_json(train_split_path)
    val_split = load_json(val_split_path)

    train_indices = [int(idx) for idx in train_split["train_indices"]]
    val_indices = {int(idx) for idx in val_split["val_indices"]}
    overlap = sorted(set(train_indices).intersection(val_indices))
    if overlap:
        raise ValueError(
            "Training split overlaps the fixed validation split; refusing to "
            f"compute embeddings. First overlapping indices: {overlap[:10]}"
        )

    device = get_device(args.device)
    encoder, transform, embedding_dim = build_encoder(args.encoder, device)
    dataset_class = DATASET_CLASSES[args.dataset]
    train_dataset = dataset_class(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )
    subset_dataset = IndexedSubsetDataset(train_dataset, train_indices)
    dataloader = DataLoader(
        subset_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    embeddings, labels, original_indices = compute_embeddings(
        model=encoder,
        dataloader=dataloader,
        device=device,
    )

    if embeddings.shape != (len(train_indices), embedding_dim):
        raise RuntimeError(
            f"Unexpected embedding shape {tuple(embeddings.shape)}; expected "
            f"({len(train_indices)}, {embedding_dim})."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = out_dir / f"embeddings_{args.encoder}.pt"
    split_hash = sha256_file(train_split_path)
    val_split_hash = sha256_file(val_split_path)
    metadata = {
        "dataset": args.dataset,
        "k": args.k,
        "subset_seed": args.subset_seed,
        "encoder": args.encoder,
        "split_hash": split_hash,
        "validation_split_hash": val_split_hash,
        "train_indices_hash": sha256_ints(train_indices),
        "split_path": str(train_split_path),
        "validation_split_path": str(val_split_path),
        "embedding_path": str(embedding_path),
        "num_train": len(train_indices),
        "embedding_dim": embedding_dim,
        "source_dataset_partition": "torchvision_train_true",
        "uses_validation_samples": False,
        "uses_test_samples": False,
        "validation_overlap_count": 0,
    }

    torch.save(
        {
            "embeddings": embeddings.float(),
            "labels": labels.long(),
            "original_indices": original_indices.long(),
            "dataset": args.dataset,
            "k": args.k,
            "subset_seed": args.subset_seed,
            "encoder": args.encoder,
            "split_hash": split_hash,
            "metadata": metadata,
        },
        embedding_path,
    )
    save_json(out_dir / "metadata.json", metadata)

    print(f"Saved embeddings: {embedding_path}")
    print(f"Saved metadata: {out_dir / 'metadata.json'}")
    print(f"Embedding shape: {tuple(embeddings.shape)}")


if __name__ == "__main__":
    main()
