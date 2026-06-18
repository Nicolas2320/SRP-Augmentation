"""Similarity-guided augmentations for SRP-Augmentation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def apply_simmixup(
    images_i: torch.Tensor,
    targets_i: torch.Tensor,
    images_j: torch.Tensor,
    targets_j: torch.Tensor,
    alpha: float = 1.0,
    mix_prob: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply SimMixUp to paired anchor and neighbor batches.

    SimMixUp uses the same linear interpolation as MixUp, but the partner
    sample is provided by a nearest-neighbor dataset rather than sampled from
    a random batch permutation.
    """
    if alpha <= 0 or mix_prob <= 0:
        return images_i, targets_i, targets_i, 1.0

    if images_i.size(0) == 0:
        return images_i, targets_i, targets_i, 1.0

    if images_i.shape != images_j.shape:
        raise ValueError(
            f"images_i and images_j must have matching shapes, got "
            f"{tuple(images_i.shape)} and {tuple(images_j.shape)}"
        )
    if targets_i.shape != targets_j.shape:
        raise ValueError(
            f"targets_i and targets_j must have matching shapes, got "
            f"{tuple(targets_i.shape)} and {tuple(targets_j.shape)}"
        )

    rng = rng or np.random.default_rng()
    if mix_prob < 1.0 and float(rng.random()) > mix_prob:
        return images_i, targets_i, targets_i, 1.0

    lam = float(rng.beta(alpha, alpha))
    mixed_images = lam * images_i + (1.0 - lam) * images_j

    return mixed_images, targets_i, targets_j, lam


def simcutmix_rand_bbox(
    size: torch.Size | tuple[int, ...],
    lam: float,
    rng: np.random.Generator,
) -> tuple[int, int, int, int]:
    """Sample a CutMix bounding box for an image batch shape [B, C, H, W]."""
    if len(size) != 4:
        raise ValueError(f"Expected BCHW image batch shape, got {tuple(size)}")

    height = int(size[2])
    width = int(size[3])
    if height <= 0 or width <= 0:
        raise ValueError(f"Image height and width must be positive, got {height}x{width}")

    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = int(rng.integers(width))
    cy = int(rng.integers(height))

    bbx1 = int(np.clip(cx - cut_w // 2, 0, width))
    bby1 = int(np.clip(cy - cut_h // 2, 0, height))
    bbx2 = int(np.clip(cx + cut_w // 2, 0, width))
    bby2 = int(np.clip(cy + cut_h // 2, 0, height))

    return bbx1, bby1, bbx2, bby2


def apply_simcutmix(
    images_i: torch.Tensor,
    targets_i: torch.Tensor,
    images_j: torch.Tensor,
    targets_j: torch.Tensor,
    alpha: float = 1.0,
    mix_prob: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply SimCutMix to paired anchor and neighbor batches.

    SimCutMix uses the same rectangular patch replacement as CutMix, but the
    partner patch is taken from the already selected neighbor batch rather than
    from a random permutation of the current batch.
    """
    if alpha <= 0 or mix_prob <= 0:
        return images_i, targets_i, targets_i, 1.0

    if images_i.size(0) == 0:
        return images_i, targets_i, targets_i, 1.0

    if images_i.shape != images_j.shape:
        raise ValueError(
            f"images_i and images_j must have matching shapes, got "
            f"{tuple(images_i.shape)} and {tuple(images_j.shape)}"
        )
    if targets_i.shape != targets_j.shape:
        raise ValueError(
            f"targets_i and targets_j must have matching shapes, got "
            f"{tuple(targets_i.shape)} and {tuple(targets_j.shape)}"
        )

    rng = rng or np.random.default_rng()
    if mix_prob < 1.0 and float(rng.random()) > mix_prob:
        return images_i, targets_i, targets_i, 1.0

    lam = float(rng.beta(alpha, alpha))
    bbx1, bby1, bbx2, bby2 = simcutmix_rand_bbox(images_i.size(), lam, rng)

    mixed_images = images_i.clone()
    mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images_j[
        :,
        :,
        bby1:bby2,
        bbx1:bbx2,
    ]

    patch_area = (bbx2 - bbx1) * (bby2 - bby1)
    image_area = images_i.size(-1) * images_i.size(-2)
    lam = 1.0 - (patch_area / image_area)

    return mixed_images, targets_i, targets_j, float(lam)


def simmixup_criterion(
    criterion: nn.Module,
    predictions: torch.Tensor,
    targets_i: torch.Tensor,
    targets_j: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute lambda-weighted cross-entropy for SimMixUp targets."""
    return lam * criterion(predictions, targets_i) + (1.0 - lam) * criterion(
        predictions,
        targets_j,
    )


def simmixup_accuracy(
    predictions: torch.Tensor,
    targets_i: torch.Tensor,
    targets_j: torch.Tensor,
    lam: float,
) -> float:
    """Compute weighted training accuracy for SimMixUp batches."""
    predicted_classes = predictions.argmax(dim=1)
    correct_i = predicted_classes.eq(targets_i).float()
    correct_j = predicted_classes.eq(targets_j).float()
    weighted_correct = lam * correct_i + (1.0 - lam) * correct_j
    return float(weighted_correct.sum().item())
