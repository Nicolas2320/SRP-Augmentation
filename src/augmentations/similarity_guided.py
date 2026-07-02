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
