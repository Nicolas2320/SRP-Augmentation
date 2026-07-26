"""Build nearest-neighbor files from k-shot training embeddings only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


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


def get_search_device(requested: str | torch.device) -> torch.device:
    """Resolve the device used for blockwise cosine-similarity search."""
    if isinstance(requested, torch.device):
        device = requested
    elif requested == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA neighbor search was requested, but CUDA is not available")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS neighbor search was requested, but MPS is not available")
    return device


def _normalized_embeddings_on_device(
    embeddings: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Move embeddings once and normalize in place without mutating the input."""
    if embeddings.device == device and embeddings.dtype == torch.float32:
        normalized = embeddings.detach().clone()
    else:
        normalized = embeddings.detach().to(device=device, dtype=torch.float32)
    norms = normalized.norm(p=2, dim=1, keepdim=True).clamp_min_(1e-12)
    normalized.div_(norms)
    return normalized


def build_neighbors(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    original_indices: torch.Tensor,
    mode: str,
    max_neighbors: int,
    query_batch_size: int = 512,
    device: str | torch.device = "cpu",
    show_progress: bool = False,
) -> dict[str, Any]:
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {tuple(embeddings.shape)}")
    if embeddings.shape[0] != labels.numel() or labels.numel() != original_indices.numel():
        raise ValueError("Embeddings, labels, and original_indices must have matching lengths")

    if query_batch_size < 1:
        raise ValueError("--query-batch-size must be at least 1")

    labels = labels.detach().cpu().long()
    original_indices = original_indices.detach().cpu().long()
    num_samples = int(labels.numel())
    neighbor_count = effective_neighbor_count(labels, mode, max_neighbors)
    if neighbor_count < 1:
        raise ValueError(
            f"Cannot build {mode} neighbors with max_neighbors={max_neighbors}; "
            "there are not enough eligible samples."
        )

    search_device = get_search_device(device)
    try:
        normalized = _normalized_embeddings_on_device(embeddings, search_device)
        device_labels = labels.to(search_device)
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            raise RuntimeError(
                "Not enough device memory to hold the normalized embeddings. "
                "Use --device cpu or an approximate-neighbor backend."
            ) from error
        raise

    effective_batch_size = min(int(query_batch_size), num_samples)
    num_query_blocks = math.ceil(num_samples / effective_batch_size)
    progress_interval = max(1, num_query_blocks // 20)
    value_blocks: list[torch.Tensor] = []
    position_blocks: list[torch.Tensor] = []

    for block_number, start in enumerate(
        range(0, num_samples, effective_batch_size),
        start=1,
    ):
        end = min(start + effective_batch_size, num_samples)
        try:
            similarities = normalized[start:end] @ normalized.T

            if mode == "class_aware":
                eligible = device_labels[start:end, None] == device_labels[None, :]
                similarities.masked_fill_(~eligible, float("-inf"))
            elif mode == "different_label":
                eligible = device_labels[start:end, None] != device_labels[None, :]
                similarities.masked_fill_(~eligible, float("-inf"))

            local_rows = torch.arange(end - start, device=search_device)
            global_rows = torch.arange(start, end, device=search_device)
            similarities[local_rows, global_rows] = float("-inf")
            values, positions = torch.topk(similarities, k=neighbor_count, dim=1)
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                raise RuntimeError(
                    "Neighbor-search block ran out of device memory. Reduce "
                    "--query-batch-size (for example, from 512 to 256)."
                ) from error
            raise

        value_blocks.append(values.cpu())
        position_blocks.append(positions.cpu())
        del similarities, values, positions, local_rows, global_rows
        if mode in {"class_aware", "different_label"}:
            del eligible

        if show_progress and (
            block_number == 1
            or block_number == num_query_blocks
            or block_number % progress_interval == 0
        ):
            print(
                f"Neighbor search: block {block_number}/{num_query_blocks} "
                f"(queries {start}-{end - 1}, device={search_device})",
                flush=True,
            )

    values = torch.cat(value_blocks, dim=0)
    neighbor_positions = torch.cat(position_blocks, dim=0)
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
        "query_batch_size": int(effective_batch_size),
        "num_query_blocks": int(num_query_blocks),
        "search_device": str(search_device),
    }


def update_metadata(metadata_path: Path, mode: str, neighbor_path: Path, payload: dict[str, Any]) -> None:
    metadata = load_json(metadata_path)
    metadata.setdefault("neighbors", {})
    metadata["neighbors"][mode] = {
        "path": str(neighbor_path),
        "requested_max_neighbors": int(payload["requested_max_neighbors"]),
        "num_neighbors": int(payload["num_neighbors"]),
        "query_batch_size": int(payload["query_batch_size"]),
        "num_query_blocks": int(payload["num_query_blocks"]),
        "search_device": str(payload["search_device"]),
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
        help="Number of query embeddings scored at once during exact search.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Similarity-search device: auto, cpu, cuda, cuda:0, or mps.",
    )
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
            query_batch_size=args.query_batch_size,
            device=args.device,
            show_progress=True,
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
