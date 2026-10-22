"""Canonical Tiny ImageNet dataset loading.

The original Tiny ImageNet layout stores training images in class folders and
the labeled official validation images in one flat folder plus an annotation
file. This loader gives both partitions the same class mapping from
``wnids.txt`` and a deterministic sample order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

from PIL import Image
from torch.utils.data import Dataset


TINY_IMAGENET_DIRNAME = "tiny-imagenet-200"
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


def resolve_tiny_imagenet_root(root: str | Path) -> Path:
    """Resolve either the dataset folder itself or its parent data directory."""
    requested = Path(root)
    candidates = (requested, requested / TINY_IMAGENET_DIRNAME)
    for candidate in candidates:
        if (candidate / "wnids.txt").is_file():
            return candidate
    raise FileNotFoundError(
        "Tiny ImageNet was not found. Expected wnids.txt in either "
        f"{requested} or {requested / TINY_IMAGENET_DIRNAME}. "
        "Extract tiny-imagenet-200.zip below the configured data root first."
    )


def read_wnids(dataset_root: Path) -> list[str]:
    wnids = [line.strip() for line in (dataset_root / "wnids.txt").read_text().splitlines()]
    if not wnids or any(not wnid for wnid in wnids):
        raise ValueError("wnids.txt must contain one non-empty class ID per line")
    if len(wnids) != len(set(wnids)):
        raise ValueError("wnids.txt contains duplicate class IDs")
    return wnids


class TinyImageNet(Dataset):
    """Tiny ImageNet train or labeled official-validation partition."""

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val"] = "train",
        transform: Callable | None = None,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("TinyImageNet split must be 'train' or 'val'")
        self.root = resolve_tiny_imagenet_root(root)
        self.split = split
        self.transform = transform
        self.classes = read_wnids(self.root)
        self.class_to_idx = {wnid: index for index, wnid in enumerate(self.classes)}
        self.samples = self._training_samples() if split == "train" else self._validation_samples()
        self.targets = [label for _, label in self.samples]

    def _training_samples(self) -> list[tuple[Path, int]]:
        samples = []
        for wnid in self.classes:
            image_dir = self.root / "train" / wnid / "images"
            if not image_dir.is_dir():
                raise FileNotFoundError(f"Missing Tiny ImageNet class directory: {image_dir}")
            paths = sorted(
                path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
            )
            if not paths:
                raise ValueError(f"No images found for Tiny ImageNet class {wnid}")
            samples.extend((path, self.class_to_idx[wnid]) for path in paths)
        return samples

    def _validation_samples(self) -> list[tuple[Path, int]]:
        annotation_path = self.root / "val" / "val_annotations.txt"
        image_dir = self.root / "val" / "images"
        if not annotation_path.is_file() or not image_dir.is_dir():
            raise FileNotFoundError(
                "Tiny ImageNet labeled validation partition is incomplete; expected "
                f"{annotation_path} and {image_dir}"
            )

        labels_by_name: dict[str, str] = {}
        for line in annotation_path.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) < 2:
                raise ValueError(f"Malformed Tiny ImageNet validation annotation: {line!r}")
            image_name, wnid = fields[:2]
            if wnid not in self.class_to_idx:
                raise ValueError(f"Validation annotation uses unknown class ID: {wnid}")
            if image_name in labels_by_name:
                raise ValueError(f"Duplicate validation annotation for {image_name}")
            labels_by_name[image_name] = wnid

        image_paths = sorted(
            path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        image_names = {path.name for path in image_paths}
        missing_annotations = sorted(image_names.difference(labels_by_name))
        missing_images = sorted(set(labels_by_name).difference(image_names))
        if missing_annotations or missing_images:
            raise ValueError(
                "Tiny ImageNet validation images and annotations disagree: "
                f"unlabeled images={missing_annotations[:5]}, missing images={missing_images[:5]}"
            )
        return [
            (path, self.class_to_idx[labels_by_name[path.name]]) for path in image_paths
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
