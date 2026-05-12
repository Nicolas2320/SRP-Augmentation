"""
MixUp augmentation for SRP-Augmentation.

MixUp creates virtual training examples by linearly combining two images
and training with a weighted combination of their labels.

Loss:
    lambda * CE(pred, y_a) + (1 - lambda) * CE(pred, y_b)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def apply_mixup(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply MixUp to one batch.

    Parameters
    ----------
    images:
        Batch of images with shape [B, C, H, W].
    targets:
        Class labels with shape [B].
    alpha:
        Beta distribution parameter. Default is 1.0.

    Returns
    -------
    mixed_images:
        Mixed batch of images.
    targets_a:
        Original labels.
    targets_b:
        Shuffled labels.
    lam:
        Mixing coefficient.
    """
    if alpha <= 0:
        return images, targets, targets, 1.0

    batch_size = images.size(0)

    if batch_size < 2:
        return images, targets, targets, 1.0

    lam = float(np.random.beta(alpha, alpha))

    permutation = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1.0 - lam) * images[permutation]
    targets_a = targets
    targets_b = targets[permutation]

    return mixed_images, targets_a, targets_b, lam


def mixup_criterion(
    criterion: nn.Module,
    predictions: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Compute the MixUp loss.

    loss = lambda * CE(pred, y_a) + (1 - lambda) * CE(pred, y_b)
    """
    return lam * criterion(predictions, targets_a) + (1.0 - lam) * criterion(
        predictions,
        targets_b,
    )


def mixup_accuracy(
    predictions: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> float:
    """
    Compute weighted training accuracy for MixUp batches.

    Validation and test accuracy should still use normal hard labels.
    """
    predicted_classes = predictions.argmax(dim=1)

    correct_a = predicted_classes.eq(targets_a).float()
    correct_b = predicted_classes.eq(targets_b).float()

    weighted_correct = lam * correct_a + (1.0 - lam) * correct_b

    return float(weighted_correct.sum().item())