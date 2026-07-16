"""Inspect and validate k-shot embedding and neighbor files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


ENCODER_CHOICES = ["resnet18_imagenet", "resnet50_imagenet"]
NEIGHBOR_MODE_CHOICES = ["class_aware", "class_agnostic", "different_label"]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_dir(output_root: Path, dataset: str, k: int, subset_seed: int) -> Path:
    return output_root / dataset / f"k{k}_seed{subset_seed}"


def resolve_neighbor_path(out_dir: Path, mode: str, max_neighbors: int, k: int) -> Path:
    candidates = [out_dir / f"neighbors_{mode}_K{max_neighbors}.pt"]
    if mode == "class_aware":
        candidates.append(out_dir / f"neighbors_{mode}_K{min(max_neighbors, k - 1)}.pt")
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(out_dir.glob(f"neighbors_{mode}_K*.pt"))
    if matches:
        return matches[-1]
    raise FileNotFoundError(f"Could not find neighbors_{mode}_K*.pt in {out_dir}")


def validate_embeddings(
    embedding_payload: dict[str, Any],
    metadata: dict[str, Any],
    train_split: dict[str, Any],
    val_split: dict[str, Any],
    split_hash: str,
) -> list[str]:
    messages: list[str] = []
    embeddings = embedding_payload["embeddings"]
    original_indices = embedding_payload["original_indices"].long()
    train_indices = torch.tensor([int(idx) for idx in train_split["train_indices"]], dtype=torch.long)
    val_indices = {int(idx) for idx in val_split["val_indices"]}

    expected_shape = (len(train_indices), int(metadata["embedding_dim"]))
    if tuple(embeddings.shape) != expected_shape:
        raise AssertionError(f"embeddings shape {tuple(embeddings.shape)} != {expected_shape}")
    messages.append(f"embeddings shape OK: {tuple(embeddings.shape)}")

    if int(metadata["num_train"]) != len(train_indices):
        raise AssertionError("metadata num_train does not match split JSON")
    if embeddings.shape[0] != len(train_indices):
        raise AssertionError("embedding row count does not match split JSON")
    messages.append(f"num_train OK: {len(train_indices)}")

    if set(original_indices.tolist()) != set(train_indices.tolist()):
        raise AssertionError("embedding original_indices do not match the k-shot train split")
    if set(original_indices.tolist()).intersection(val_indices):
        raise AssertionError("validation indices are present in embeddings")
    messages.append("validation exclusion OK")

    if metadata.get("uses_test_samples") is not False:
        raise AssertionError("metadata does not explicitly record uses_test_samples=false")
    if metadata.get("source_dataset_partition") != "torchvision_train_true":
        raise AssertionError("metadata does not record torchvision train partition as the source")
    messages.append("test exclusion metadata OK")

    if metadata.get("split_hash") != split_hash:
        raise AssertionError("metadata split_hash does not match the current split file")
    for key in ("dataset", "k", "subset_seed", "encoder", "split_hash"):
        if key not in metadata:
            raise AssertionError(f"metadata missing required key: {key}")
    messages.append("metadata provenance OK")
    return messages


def validate_neighbors(neighbor_payload: dict[str, Any], mode: str) -> list[str]:
    messages: list[str] = []
    original_indices = neighbor_payload["original_indices"].long()
    labels = neighbor_payload["labels"].long()
    neighbor_indices = neighbor_payload["neighbor_indices"].long()
    neighbor_labels = neighbor_payload["neighbor_labels"].long()
    similarities = neighbor_payload["similarities"].float()

    if (neighbor_indices == original_indices[:, None]).any():
        raise AssertionError("at least one sample is its own neighbor")
    messages.append("self-neighbor exclusion OK")

    if mode == "class_aware" and (neighbor_labels != labels[:, None]).any():
        raise AssertionError("class-aware neighbors include a different label")
    if mode == "class_aware":
        messages.append("class-aware labels OK")

    if mode == "different_label" and (neighbor_labels == labels[:, None]).any():
        raise AssertionError("different-label neighbors include a matching label")
    if mode == "different_label":
        messages.append("different-label labels OK")

    if mode == "class_agnostic":
        mixed_count = int((neighbor_labels != labels[:, None]).sum().item())
        messages.append(f"class-agnostic mixed-label neighbor count: {mixed_count}")

    if similarities.shape[1] > 1:
        if (similarities[:, :-1] < similarities[:, 1:]).any():
            raise AssertionError("cosine similarity values are not sorted descending")
    messages.append("similarity sorting OK")
    return messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate embedding and neighbor artifacts.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--subset-seed", type=int, required=True)
    parser.add_argument("--encoder", choices=ENCODER_CHOICES, required=True)
    parser.add_argument("--mode", choices=NEIGHBOR_MODE_CHOICES, required=True)
    parser.add_argument("--max-neighbors", type=int, required=True)
    parser.add_argument("--split-root", type=str, default="data/splits")
    parser.add_argument("--output-root", type=str, default="results/neighbors")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = output_dir(Path(args.output_root), args.dataset, args.k, args.subset_seed)
    embedding_path = out_dir / f"embeddings_{args.encoder}.pt"
    neighbor_path = resolve_neighbor_path(out_dir, args.mode, args.max_neighbors, args.k)
    metadata_path = out_dir / "metadata.json"
    train_split_path = Path(args.split_root) / args.dataset / f"k{args.k}_seed{args.subset_seed}.json"
    val_split_path = Path(args.split_root) / args.dataset / "fixed_validation_split.json"

    embedding_payload = torch.load(embedding_path, map_location="cpu")
    neighbor_payload = torch.load(neighbor_path, map_location="cpu")
    metadata = load_json(metadata_path)
    train_split = load_json(train_split_path)
    val_split = load_json(val_split_path)
    split_hash = sha256_file(train_split_path)

    messages = []
    messages.extend(validate_embeddings(embedding_payload, metadata, train_split, val_split, split_hash))
    messages.extend(validate_neighbors(neighbor_payload, args.mode))

    print(f"Validated embeddings: {embedding_path}")
    print(f"Validated neighbors: {neighbor_path}")
    for message in messages:
        print(f"- {message}")


if __name__ == "__main__":
    main()
