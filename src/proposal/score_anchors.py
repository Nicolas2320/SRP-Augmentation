"""Score k-shot training anchors for targeted guided augmentation.

The script combines two signals:

- uncertainty: normalized model entropy on each clean training anchor
- rarity: distance from the sample embedding to its class center

Both signals are converted to percentiles before weighting so one scale cannot
accidentally dominate the other.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from train import (  # noqa: E402
    build_model,
    get_dataset_class,
    get_num_classes,
    get_split_paths,
    get_transforms,
    load_json,
)


class IndexedSubsetDataset(Dataset):
    """Return CIFAR samples and their original training-set indices."""

    def __init__(self, base_dataset: Dataset, indices: list[int]) -> None:
        self.base_dataset = base_dataset
        self.indices = [int(index) for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int, int]:
        original_index = self.indices[position]
        image, label = self.base_dataset[original_index]
        return image, int(label), original_index


def get_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


@torch.no_grad()
def score_uncertainty(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    entropies: list[torch.Tensor] = []
    losses: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    original_indices: list[torch.Tensor] = []

    model.eval()
    for images, batch_labels, batch_indices in dataloader:
        images = images.to(device)
        batch_labels = batch_labels.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).clamp_min(1e-12)
        entropy = -(probs * probs.log()).sum(dim=1) / math.log(num_classes)
        loss = F.cross_entropy(logits, batch_labels, reduction="none")

        entropies.append(entropy.cpu())
        losses.append(loss.cpu())
        labels.append(batch_labels.cpu().long())
        original_indices.append(batch_indices.cpu().long())

    return (
        torch.cat(entropies),
        torch.cat(losses),
        torch.cat(labels),
        torch.cat(original_indices),
    )


def compute_rarity(
    embedding_payload: dict[str, Any],
    original_indices: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    required = {"embeddings", "labels", "original_indices"}
    missing = sorted(required.difference(embedding_payload))
    if missing:
        raise ValueError(f"Embedding file is missing required keys: {missing}")

    embeddings = F.normalize(torch.as_tensor(embedding_payload["embeddings"]).float(), dim=1)
    embedding_labels = torch.as_tensor(embedding_payload["labels"], dtype=torch.long)
    embedding_indices = torch.as_tensor(embedding_payload["original_indices"], dtype=torch.long)
    row_by_index = {int(index): row for row, index in enumerate(embedding_indices.tolist())}

    missing_indices = [int(index) for index in original_indices.tolist() if int(index) not in row_by_index]
    if missing_indices:
        raise ValueError(f"Embedding file is missing train indices: {missing_indices[:10]}")

    rows = torch.tensor([row_by_index[int(index)] for index in original_indices.tolist()])
    aligned_embeddings = embeddings[rows]
    aligned_labels = embedding_labels[rows]
    if not torch.equal(aligned_labels, labels):
        raise ValueError("Embedding labels do not align with dataset labels")

    centers = torch.zeros((num_classes, aligned_embeddings.shape[1]), dtype=aligned_embeddings.dtype)
    for class_id in range(num_classes):
        mask = labels == class_id
        if not bool(mask.any().item()):
            raise ValueError(f"No samples found for class {class_id}")
        centers[class_id] = aligned_embeddings[mask].mean(dim=0)

    distances = torch.norm(aligned_embeddings - centers[labels], dim=1)
    rarity_percentiles = torch.empty_like(distances)
    for class_id in range(num_classes):
        mask = labels == class_id
        rarity_percentiles[mask] = rank_percentile(distances[mask])

    return distances, rarity_percentiles


def output_name(args: argparse.Namespace) -> str:
    checkpoint = Path(args.checkpoint_path).stem
    embedding = Path(args.embedding_path).stem
    uncertainty_weight = format_float(args.uncertainty_weight)
    rarity_weight = format_float(args.rarity_weight)
    return (
        f"{args.dataset}_{args.model}_k{args.k}_seed{args.subset_seed}_"
        f"{checkpoint}_{embedding}_uw{uncertainty_weight}_rw{rarity_weight}"
    )


def format_float(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def save_csv(path: Path, payload: dict[str, Any]) -> None:
    fieldnames = [
        "original_index",
        "label",
        "uncertainty",
        "uncertainty_percentile",
        "loss",
        "rarity",
        "rarity_percentile",
        "score",
        "score_percentile",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in range(payload["original_indices"].numel()):
            writer.writerow(
                {
                    "original_index": int(payload["original_indices"][row].item()),
                    "label": int(payload["labels"][row].item()),
                    "uncertainty": float(payload["uncertainty"][row].item()),
                    "uncertainty_percentile": float(payload["uncertainty_percentile"][row].item()),
                    "loss": float(payload["loss"][row].item()),
                    "rarity": float(payload["rarity"][row].item()),
                    "rarity_percentile": float(payload["rarity_percentile"][row].item()),
                    "score": float(payload["score"][row].item()),
                    "score_percentile": float(payload["score_percentile"][row].item()),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score k-shot anchors for targeted augmentation.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--model", choices=["resnet50", "vit"], required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--subset-seed", type=int, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--embedding-path", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--split-root", type=str, default="data/splits")
    parser.add_argument("--output-root", type=str, default="results/anchor_scores")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--uncertainty-weight", type=float, default=0.7)
    parser.add_argument("--rarity-weight", type=float, default=0.3)
    parser.add_argument("--output-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.uncertainty_weight < 0 or args.rarity_weight < 0:
        raise ValueError("Score weights must be non-negative")
    weight_sum = args.uncertainty_weight + args.rarity_weight
    if weight_sum <= 0:
        raise ValueError("At least one score weight must be positive")

    train_split_path, _ = get_split_paths(
        dataset=args.dataset,
        k=args.k,
        subset_seed=args.subset_seed,
        split_root=args.split_root,
    )
    train_indices = [int(index) for index in load_json(train_split_path)["train_indices"]]
    num_classes = get_num_classes(args.dataset)
    device = get_device(args.device)

    _, eval_transform = get_transforms(args.dataset, augmentation="none", seed=0)
    dataset_class = get_dataset_class(args.dataset)
    base_dataset = dataset_class(
        root=args.data_root,
        train=True,
        download=False,
        transform=eval_transform,
    )
    dataloader = DataLoader(
        IndexedSubsetDataset(base_dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.model, num_classes=num_classes).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    uncertainty, loss, labels, original_indices = score_uncertainty(
        model=model,
        dataloader=dataloader,
        device=device,
        num_classes=num_classes,
    )
    embedding_payload = torch.load(args.embedding_path, map_location="cpu")
    rarity, rarity_percentile = compute_rarity(
        embedding_payload=embedding_payload,
        original_indices=original_indices,
        labels=labels,
        num_classes=num_classes,
    )

    uncertainty_percentile = rank_percentile(uncertainty)
    normalized_uncertainty_weight = args.uncertainty_weight / weight_sum
    normalized_rarity_weight = args.rarity_weight / weight_sum
    score = (
        normalized_uncertainty_weight * uncertainty_percentile
        + normalized_rarity_weight * rarity_percentile
    )
    score_percentile = rank_percentile(score)

    payload = {
        "original_indices": original_indices.long(),
        "labels": labels.long(),
        "uncertainty": uncertainty.float(),
        "uncertainty_percentile": uncertainty_percentile.float(),
        "loss": loss.float(),
        "rarity": rarity.float(),
        "rarity_percentile": rarity_percentile.float(),
        "score": score.float(),
        "score_percentile": score_percentile.float(),
        "metadata": {
            "dataset": args.dataset,
            "model": args.model,
            "k": args.k,
            "subset_seed": args.subset_seed,
            "checkpoint_path": args.checkpoint_path,
            "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
            "embedding_path": args.embedding_path,
            "uncertainty_weight": normalized_uncertainty_weight,
            "rarity_weight": normalized_rarity_weight,
            "num_train": len(train_indices),
        },
    }

    out_dir = Path(args.output_root) / args.dataset / f"k{args.k}_seed{args.subset_seed}"
    name = args.output_name or output_name(args)
    score_path = out_dir / f"{name}.pt"
    csv_path = out_dir / f"{name}.csv"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, score_path)
    save_csv(csv_path, payload)

    print(f"Saved anchor scores: {score_path}")
    print(f"Saved anchor score CSV: {csv_path}")
    print(
        json.dumps(
            {
                "num_train": len(train_indices),
                "mean_uncertainty": float(uncertainty.mean().item()),
                "mean_rarity_percentile": float(rarity_percentile.mean().item()),
                "mean_score": float(score.mean().item()),
                "top10_score_threshold": float(torch.quantile(score, 0.9).item()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
