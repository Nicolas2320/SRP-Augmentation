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


def get_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    query_batch_size: int = 512,
) -> dict[str, Any]:
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {tuple(embeddings.shape)}")
    if embeddings.shape[0] != labels.numel() or labels.numel() != original_indices.numel():
        raise ValueError("Embeddings, labels, and original_indices must have matching lengths")
    if query_batch_size < 1:
        raise ValueError("query_batch_size must be at least 1")

    labels = labels.long()
    labels_cpu = labels.cpu()
    original_indices = original_indices.long().cpu()
    num_samples = int(labels.numel())
    neighbor_count = effective_neighbor_count(labels, mode, max_neighbors)
    if neighbor_count < 1:
        raise ValueError(
            f"Cannot build {mode} neighbors with max_neighbors={max_neighbors}; "
            "there are not enough eligible samples."
        )

    normalized = F.normalize(embeddings.float(), dim=1)
    value_batches: list[torch.Tensor] = []
    position_batches: list[torch.Tensor] = []

    # Avoid materializing an N x N similarity matrix. At CIFAR-100 k=450,
    # N=45,000 and the dense matrix alone would require about 8.1 GB in
    # float32, before masks and other intermediate tensors are included.
    for start in range(0, num_samples, query_batch_size):
        end = min(start + query_batch_size, num_samples)
        similarities = normalized[start:end] @ normalized.T

        local_rows = torch.arange(end - start, device=similarities.device)
        global_rows = torch.arange(start, end, device=similarities.device)
        similarities[local_rows, global_rows] = float("-inf")

        if mode == "class_aware":
            eligible = labels[start:end, None] == labels[None, :]
            similarities.masked_fill_(~eligible, float("-inf"))
        elif mode == "different_label":
            eligible = labels[start:end, None] != labels[None, :]
            similarities.masked_fill_(~eligible, float("-inf"))

        batch_values, batch_positions = torch.topk(
            similarities,
            k=neighbor_count,
            dim=1,
        )
        value_batches.append(batch_values.cpu())
        position_batches.append(batch_positions.cpu())

    values = torch.cat(value_batches, dim=0)
    neighbor_positions = torch.cat(position_batches, dim=0)
    if not torch.isfinite(values).all():
        raise RuntimeError("At least one sample has fewer eligible neighbors than requested")

    return {
        "mode": mode,
        "requested_max_neighbors": int(max_neighbors),
        "num_neighbors": int(neighbor_count),
        "query_batch_size": int(query_batch_size),
        "similarities": values.float(),
        "neighbor_positions": neighbor_positions.long(),
        "neighbor_indices": original_indices[neighbor_positions].long(),
        "neighbor_labels": labels_cpu[neighbor_positions].long(),
        "original_indices": original_indices.long(),
        "labels": labels_cpu.long(),
    }


def update_metadata(metadata_path: Path, mode: str, neighbor_path: Path, payload: dict[str, Any]) -> None:
    metadata = load_json(metadata_path)
    metadata.setdefault("neighbors", {})
    metadata["neighbors"][mode] = {
        "path": str(neighbor_path),
        "requested_max_neighbors": int(payload["requested_max_neighbors"]),
        "num_neighbors": int(payload["num_neighbors"]),
        "query_batch_size": int(payload["query_batch_size"]),
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
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=512,
        help="Number of anchor embeddings processed per similarity block.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Similarity-computation device: auto, cpu, cuda, or another torch device.",
    )
    parser.add_argument("--output-root", type=str, default="results/experiments/shared/neighbors")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
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
            embeddings=embedding_payload["embeddings"].to(device),
            labels=embedding_payload["labels"].to(device),
            original_indices=embedding_payload["original_indices"],
            mode=mode,
            max_neighbors=args.max_neighbors,
            query_batch_size=args.query_batch_size,
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
            f"(requested={args.max_neighbors}, actual={neighbor_payload['num_neighbors']}, "
            f"device={device})"
        )


if __name__ == "__main__":
    main()
