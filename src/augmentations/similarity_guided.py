"""Similarity-guided augmentations for SRP-Augmentation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sample_uniform(rng: np.random.Generator, size: int) -> np.ndarray:
    try:
        return np.asarray(rng.random(size), dtype=np.float32)
    except TypeError:
        return np.asarray([rng.random() for _ in range(size)], dtype=np.float32)


def _sample_mix_mask(
    batch_size: int,
    mix_prob: float,
    sample_mix_prob: torch.Tensor | None,
    device: torch.device,
    rng: np.random.Generator,
) -> torch.Tensor | None:
    if sample_mix_prob is None:
        return None

    sample_probs = torch.as_tensor(sample_mix_prob, dtype=torch.float32, device=device)
    if sample_probs.ndim != 1 or sample_probs.numel() != batch_size:
        raise ValueError(
            "sample_mix_prob must be a 1D tensor with one value per batch item, "
            f"got shape {tuple(sample_probs.shape)} for batch size {batch_size}"
        )

    probabilities = torch.clamp(sample_probs * float(mix_prob), min=0.0, max=1.0)
    draws = torch.as_tensor(_sample_uniform(rng, batch_size), dtype=torch.float32, device=device)
    return draws < probabilities


def _lambda_for_mask(mask: torch.Tensor, lam: float) -> torch.Tensor:
    lam_tensor = torch.ones(mask.numel(), dtype=torch.float32, device=mask.device)
    lam_tensor[mask] = float(lam)
    return lam_tensor


def apply_simmixup(
    images_i: torch.Tensor,
    targets_i: torch.Tensor,
    images_j: torch.Tensor,
    targets_j: torch.Tensor,
    alpha: float = 1.0,
    mix_prob: float = 1.0,
    sample_mix_prob: torch.Tensor | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float | torch.Tensor]:
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
    mix_mask = _sample_mix_mask(
        batch_size=images_i.size(0),
        mix_prob=mix_prob,
        sample_mix_prob=sample_mix_prob,
        device=images_i.device,
        rng=rng,
    )
    if mix_mask is None and mix_prob < 1.0 and float(rng.random()) > mix_prob:
        return images_i, targets_i, targets_i, 1.0
    if mix_mask is not None and not bool(mix_mask.any().item()):
        return images_i, targets_i, targets_j, _lambda_for_mask(mix_mask, 1.0)

    lam = float(rng.beta(alpha, alpha))
    if mix_mask is None:
        mixed_images = lam * images_i + (1.0 - lam) * images_j
        return mixed_images, targets_i, targets_j, lam

    lam_tensor = _lambda_for_mask(mix_mask, lam)
    view_shape = (lam_tensor.numel(),) + (1,) * (images_i.ndim - 1)
    mix_weight = lam_tensor.view(view_shape)
    mixed_images = mix_weight * images_i + (1.0 - mix_weight) * images_j

    return mixed_images, targets_i, targets_j, lam_tensor


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
    sample_mix_prob: torch.Tensor | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float | torch.Tensor]:
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
    mix_mask = _sample_mix_mask(
        batch_size=images_i.size(0),
        mix_prob=mix_prob,
        sample_mix_prob=sample_mix_prob,
        device=images_i.device,
        rng=rng,
    )
    if mix_mask is None and mix_prob < 1.0 and float(rng.random()) > mix_prob:
        return images_i, targets_i, targets_i, 1.0
    if mix_mask is not None and not bool(mix_mask.any().item()):
        return images_i, targets_i, targets_j, _lambda_for_mask(mix_mask, 1.0)

    lam = float(rng.beta(alpha, alpha))
    bbx1, bby1, bbx2, bby2 = simcutmix_rand_bbox(images_i.size(), lam, rng)

    mixed_images = images_i.clone()
    if mix_mask is None:
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images_j[
            :,
            :,
            bby1:bby2,
            bbx1:bbx2,
        ]
    else:
        mixed_images[mix_mask, :, bby1:bby2, bbx1:bbx2] = images_j[
            mix_mask,
            :,
            bby1:bby2,
            bbx1:bbx2,
        ]

    patch_area = (bbx2 - bbx1) * (bby2 - bby1)
    image_area = images_i.size(-1) * images_i.size(-2)
    lam = 1.0 - (patch_area / image_area)

    if mix_mask is None:
        return mixed_images, targets_i, targets_j, float(lam)

    return mixed_images, targets_i, targets_j, _lambda_for_mask(mix_mask, float(lam))


def _cross_entropy_per_sample(
    criterion: nn.Module,
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if isinstance(criterion, nn.CrossEntropyLoss):
        return F.cross_entropy(
            predictions,
            targets,
            weight=criterion.weight,
            ignore_index=criterion.ignore_index,
            reduction="none",
            label_smoothing=criterion.label_smoothing,
        )
    return F.cross_entropy(predictions, targets, reduction="none")


def simmixup_criterion(
    criterion: nn.Module,
    predictions: torch.Tensor,
    targets_i: torch.Tensor,
    targets_j: torch.Tensor,
    lam: float | torch.Tensor,
) -> torch.Tensor:
    """Compute lambda-weighted cross-entropy for SimMixUp targets."""
    if torch.is_tensor(lam):
        lam = lam.to(device=predictions.device, dtype=predictions.dtype)
        loss_i = _cross_entropy_per_sample(criterion, predictions, targets_i)
        loss_j = _cross_entropy_per_sample(criterion, predictions, targets_j)
        return (lam * loss_i + (1.0 - lam) * loss_j).mean()

    return lam * criterion(predictions, targets_i) + (1.0 - lam) * criterion(
        predictions,
        targets_j,
    )


def simmixup_accuracy(
    predictions: torch.Tensor,
    targets_i: torch.Tensor,
    targets_j: torch.Tensor,
    lam: float | torch.Tensor,
) -> float:
    """Compute weighted training accuracy for SimMixUp batches."""
    predicted_classes = predictions.argmax(dim=1)
    correct_i = predicted_classes.eq(targets_i).float()
    correct_j = predicted_classes.eq(targets_j).float()
    if torch.is_tensor(lam):
        lam = lam.to(device=predictions.device, dtype=correct_i.dtype)
    weighted_correct = lam * correct_i + (1.0 - lam) * correct_j
    return float(weighted_correct.sum().item())
