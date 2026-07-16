"""Build nearest-neighbor files from k-shot training embeddings only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ENCODER_CHOICES = ["resnet18_imagenet", "resnet50_imagenet"]
NEIGHBOR_MODE_CHOICES = ["class_aware", "class_agnostic", "different_label"]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def output_dir(output_root: Path, dataset: str, k: int, subset_seed: int) -> Path:
    return output_root / dataset / f"k{k}_seed{subset_seed}"


def load_embedding_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {path}")
    payload = torch.load(path, map_location="cpu")
    required = {"embeddings", "labels", "original_indices", "dataset", "k", "subset_seed", "encoder"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Embeddings file is missing required keys: {missing}")
    return payload


def effective_neighbor_count(labels: torch.Tensor, mode: str, max_neighbors: int) -> int:
    if max_neighbors < 1:
        raise ValueError("--max-neighbors must be at least 1")

    if mode == "class_agnostic":
        return min(max_neighbors, int(labels.numel()) - 1)

    if mode == "different_label":
        _, counts = torch.unique(labels, return_counts=True)
        largest_class_count = int(counts.max().item())
        return min(max_neighbors, int(labels.numel()) - largest_class_count)

    if mode != "class_aware":
        raise ValueError(f"Unsupported neighbor mode: {mode}")

    _, counts = torch.unique(labels, return_counts=True)
    smallest_class_count = int(counts.min().item())
    return min(max_neighbors, smallest_class_count - 1)


def build_neighbors(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    original_indices: torch.Tensor,
    mode: str,
    max_neighbors: int,
) -> dict[str, Any]:
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {tuple(embeddings.shape)}")
    if embeddings.shape[0] != labels.numel() or labels.numel() != original_indices.numel():
        raise ValueError("Embeddings, labels, and original_indices must have matching lengths")

    labels = labels.long()
    original_indices = original_indices.long()
    num_samples = int(labels.numel())
    neighbor_count = effective_neighbor_count(labels, mode, max_neighbors)
    if neighbor_count < 1:
        raise ValueError(
            f"Cannot build {mode} neighbors with max_neighbors={max_neighbors}; "
            "there are not enough eligible samples."
        )

    normalized = F.normalize(embeddings.float(), dim=1)
    similarities = normalized @ normalized.T
    eligible = torch.ones((num_samples, num_samples), dtype=torch.bool)
    eligible.fill_diagonal_(False)
    if mode == "class_aware":
        eligible &= labels[:, None] == labels[None, :]
    elif mode == "different_label":
        eligible &= labels[:, None] != labels[None, :]

    masked_similarities = similarities.masked_fill(~eligible, float("-inf"))
    values, neighbor_positions = torch.topk(masked_similarities, k=neighbor_count, dim=1)
    if not torch.isfinite(values).all():
        raise RuntimeError("At least one sample has fewer eligible neighbors than requested")

    return {
        "mode": mode,
        "requested_max_neighbors": int(max_neighbors),
        "num_neighbors": int(neighbor_count),
        "similarities": values.float(),
        "neighbor_positions": neighbor_positions.long(),
        "neighbor_indices": original_indices[neighbor_positions].long(),
        "neighbor_labels": labels[neighbor_positions].long(),
        "original_indices": original_indices.long(),
        "labels": labels.long(),
    }


def update_metadata(metadata_path: Path, mode: str, neighbor_path: Path, payload: dict[str, Any]) -> None:
    metadata = load_json(metadata_path)
    metadata.setdefault("neighbors", {})
    metadata["neighbors"][mode] = {
        "path": str(neighbor_path),
        "requested_max_neighbors": int(payload["requested_max_neighbors"]),
        "num_neighbors": int(payload["num_neighbors"]),
    }
    save_json(metadata_path, metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build filtered nearest-neighbor files from saved embeddings."
    )
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--subset-seed", type=int, required=True)
    parser.add_argument("--encoder", choices=ENCODER_CHOICES, required=True)
    parser.add_argument("--mode", choices=NEIGHBOR_MODE_CHOICES + ["both"], required=True)
    parser.add_argument("--max-neighbors", type=int, required=True)
    parser.add_argument("--output-root", type=str, default="results/experiments/shared/neighbors")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = output_dir(Path(args.output_root), args.dataset, args.k, args.subset_seed)
    embedding_path = out_dir / f"embeddings_{args.encoder}.pt"
    embedding_payload = load_embedding_payload(embedding_path)

    for key in ("dataset", "k", "subset_seed", "encoder"):
        if embedding_payload[key] != getattr(args, key.replace("-", "_"), None):
            raise ValueError(
                f"Embeddings payload has {key}={embedding_payload[key]!r}, "
                f"but command requested {getattr(args, key)!r}."
            )

    modes = ["class_aware", "class_agnostic"] if args.mode == "both" else [args.mode]
    for mode in modes:
        neighbor_payload = build_neighbors(
            embeddings=embedding_payload["embeddings"],
            labels=embedding_payload["labels"],
            original_indices=embedding_payload["original_indices"],
            mode=mode,
            max_neighbors=args.max_neighbors,
        )
        neighbor_payload.update(
            {
                "dataset": args.dataset,
                "k": args.k,
                "subset_seed": args.subset_seed,
                "encoder": args.encoder,
                "split_hash": embedding_payload.get("split_hash"),
                "embedding_path": str(embedding_path),
            }
        )
        neighbor_path = out_dir / f"neighbors_{mode}_K{neighbor_payload['num_neighbors']}.pt"
        torch.save(neighbor_payload, neighbor_path)
        update_metadata(out_dir / "metadata.json", mode, neighbor_path, neighbor_payload)
        print(
            f"Saved {mode} neighbors: {neighbor_path} "
            f"(requested={args.max_neighbors}, actual={neighbor_payload['num_neighbors']})"
        )


if __name__ == "__main__":
    main()
