"""Combined AugMix implementation.

This script contains two separated sections:

1. Base augmentation operators
2. AugMix data augmentation method
Source : https://github.com/google-research/augmix
It also includes a small self-test that runs AugMix on a random CIFAR-sized image.
"""

# ============================================================
# 1. Base augmentation operators
# ============================================================

import numpy as np
from PIL import Image, ImageOps, ImageEnhance

# CIFAR image size. For ImageNet, change this value accordingly.
IMAGE_SIZE = 32


def int_parameter(level, maxval):
    """Scale `maxval` according to `level` and return an integer.

    Args:
        level: Operation level, normally between 0 and 10.
        maxval: Maximum value that the operation can have.

    Returns:
        Integer-scaled operation value.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Scale `maxval` according to `level` and return a float.

    Args:
        level: Operation level, normally between 0 and 10.
        maxval: Maximum value that the operation can have.

    Returns:
        Float-scaled operation value.
    """
    return float(level) * maxval / 10.0


def sample_level(n, rng):
    """Sample an augmentation level uniformly from [0.1, n)."""
    return rng.uniform(low=0.1, high=n)


def autocontrast(pil_img, _, rng):
    """Apply autocontrast."""
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _, rng):
    """Apply histogram equalization."""
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level, rng):
    """Reduce the number of bits for each color channel."""
    level = int_parameter(sample_level(level, rng), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level, rng):
    """Rotate the image by a random degree up to 30 degrees."""
    degrees = int_parameter(sample_level(level, rng), 30)
    if rng.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level, rng):
    """Invert all pixel values above a threshold."""
    level = int_parameter(sample_level(level, rng), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level, rng):
    """Shear the image along the x-axis."""
    level = float_parameter(sample_level(level, rng), 0.3)
    if rng.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.AFFINE,
        (1, level, 0, 0, 1, 0),
        resample=Image.BILINEAR,
    )


def shear_y(pil_img, level, rng):
    """Shear the image along the y-axis."""
    level = float_parameter(sample_level(level, rng), 0.3)
    if rng.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.AFFINE,
        (1, 0, 0, level, 1, 0),
        resample=Image.BILINEAR,
    )


def translate_x(pil_img, level, rng):
    """Translate the image along the x-axis."""
    level = int_parameter(sample_level(level, rng), IMAGE_SIZE / 3)
    if rng.random() > 0.5:
        level = -level
    return pil_img.transform(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.AFFINE,
        (1, 0, level, 0, 1, 0),
        resample=Image.BILINEAR,
    )


def translate_y(pil_img, level, rng):
    """Translate the image along the y-axis."""
    level = int_parameter(sample_level(level, rng), IMAGE_SIZE / 3)
    if rng.random() > 0.5:
        level = -level
    return pil_img.transform(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.AFFINE,
        (1, 0, 0, 0, 1, level),
        resample=Image.BILINEAR,
    )


# Operations that overlap with ImageNet-C's test set
def color(pil_img, level, rng):
    """Adjust image color balance."""
    level = float_parameter(sample_level(level, rng), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


def contrast(pil_img, level, rng):
    """Adjust image contrast."""
    level = float_parameter(sample_level(level, rng), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


def brightness(pil_img, level, rng):
    """Adjust image brightness."""
    level = float_parameter(sample_level(level, rng), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


def sharpness(pil_img, level, rng):
    """Adjust image sharpness."""
    level = float_parameter(sample_level(level, rng), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


# Standard AugMix augmentation set
augmentations = [
    autocontrast,
    equalize,
    posterize,
    rotate,
    solarize,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
]

# Extended augmentation set
augmentations_all = [
    autocontrast,
    equalize,
    posterize,
    rotate,
    solarize,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
    color,
    contrast,
    brightness,
    sharpness,
]


# ============================================================
# 2. AugMix data augmentation method
# ============================================================

def normalize(image, mean, std):
    """Normalize input image channel-wise to zero mean and unit variance.

    Args:
        image: Input image as float ndarray of shape (height, width, channels),
               with values in [0, 1].

    Returns:
        Normalized image with the same shape as the input.
    """
    image = image.transpose(2, 0, 1)  # Switch to channel-first
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    image = (image - mean[:, None, None]) / std[:, None, None]
    return image.transpose(1, 2, 0).astype(np.float32, copy=False)


def apply_op(image, op, severity, rng):
    """Apply one augmentation operation to an image.

    Args:
        image: Float ndarray image with values in [0, 1].
        op: Augmentation function that accepts a PIL image and severity level.
        severity: Strength of the augmentation.

    Returns:
        Augmented image as float ndarray with values in [0, 1].
    """
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(image)
    pil_img = op(pil_img, severity, rng)
    return np.asarray(pil_img) / 255.0


def augment_and_mix(
    image,
    mean,
    std,
    severity=3,
    width=3,
    depth=-1,
    alpha=1.0,
    rng=None,
):
    """Perform AugMix augmentation and compute the final mixture.

    Args:
        image: Raw input image as float32 ndarray of shape (h, w, c),
               with values in [0, 1].
        severity: Severity of augmentation operators, usually between 1 and 10.
        width: Number of augmentation chains.
        depth: Number of operations per chain. If -1, depth is sampled uniformly
               from {1, 2, 3}.
        alpha: Probability coefficient for Beta and Dirichlet distributions.
        rng: Seeded NumPy generator used for all random AugMix choices.

    Returns:
        Augmented and mixed normalized image.
    """
    rng = rng or np.random.default_rng()
    ws = np.float32(rng.dirichlet([alpha] * width))
    m = np.float32(rng.beta(alpha, alpha))

    mix = np.zeros_like(image, dtype=np.float32)

    for i in range(width):
        image_aug = image.copy()
        d = depth if depth > 0 else rng.integers(1, 4)

        for _ in range(d):
            op = rng.choice(augmentations)
            image_aug = apply_op(image_aug, op, severity, rng)

        # Preprocessing commutes because all coefficients are convex.
        mix += ws[i] * normalize(image_aug, mean, std)

    mixed = (1 - m) * normalize(image, mean, std) + m * mix
    return mixed


class AugMixTransform:
    """
    Torchvision-compatible AugMix transform.

    Input:
        PIL image from CIFAR dataset.

    Output:
        Normalized torch.Tensor with shape [C, H, W].
    """

    def __init__(
        self,
        mean,
        std,
        severity: int = 3,
        width: int = 3,
        depth: int = -1,
        alpha: float = 1.0,
        seed: int | None = None,
    ):
        self.mean = mean
        self.std = std
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

    def set_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, pil_img):
        import torch

        image = np.asarray(pil_img).astype(np.float32) / 255.0

        mixed = augment_and_mix(
            image,
            mean=self.mean,
            std=self.std,
            severity=self.severity,
            width=self.width,
            depth=self.depth,
            alpha=self.alpha,
            rng=self.rng,
        )

        tensor = torch.from_numpy(mixed).permute(2, 0, 1).float()
        return tensor



# ============================================================
# 3. Self-test
# ============================================================

def _self_test():
    """Run a minimal test to confirm that AugMix works."""
    rng = np.random.default_rng(42)

    # Create a random CIFAR-10-like image in [0, 1].
    image = rng.random((IMAGE_SIZE, IMAGE_SIZE, 3)).astype(np.float32)

    mixed = augment_and_mix(
        image,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
        severity=3,
        width=3,
        depth=-1,
        alpha=1.0,
        rng=rng,
    )

    assert isinstance(mixed, np.ndarray), "Output must be a NumPy array."
    assert mixed.shape == image.shape, f"Expected shape {image.shape}, got {mixed.shape}."
    assert np.isfinite(mixed).all(), "Output contains NaN or infinite values."

    print("AugMix self-test passed.")
    print(f"Input shape: {image.shape}")
    print(f"Output shape: {mixed.shape}")
    print(f"Output dtype: {mixed.dtype}")
    print(f"Output min: {mixed.min():.4f}")
    print(f"Output max: {mixed.max():.4f}")


if __name__ == "__main__":
    _self_test()
