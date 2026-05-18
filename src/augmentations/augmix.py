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

# CIFAR-10 image size. For ImageNet, change this value accordingly.
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


def sample_level(n):
    """Sample an augmentation level uniformly from [0.1, n)."""
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
    """Apply autocontrast."""
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    """Apply histogram equalization."""
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    """Reduce the number of bits for each color channel."""
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    """Rotate the image by a random degree up to 30 degrees."""
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    """Invert all pixel values above a threshold."""
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    """Shear the image along the x-axis."""
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.AFFINE,
        (1, level, 0, 0, 1, 0),
        resample=Image.BILINEAR,
    )


def shear_y(pil_img, level):
    """Shear the image along the y-axis."""
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.AFFINE,
        (1, 0, 0, level, 1, 0),
        resample=Image.BILINEAR,
    )


def translate_x(pil_img, level):
    """Translate the image along the x-axis."""
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.AFFINE,
        (1, 0, level, 0, 1, 0),
        resample=Image.BILINEAR,
    )


def translate_y(pil_img, level):
    """Translate the image along the y-axis."""
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.AFFINE,
        (1, 0, 0, 0, 1, level),
        resample=Image.BILINEAR,
    )


# Operations that overlap with ImageNet-C's test set
def color(pil_img, level):
    """Adjust image color balance."""
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


def contrast(pil_img, level):
    """Adjust image contrast."""
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


def brightness(pil_img, level):
    """Adjust image brightness."""
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


def sharpness(pil_img, level):
    """Adjust image sharpness."""
    level = float_parameter(sample_level(level), 1.8) + 0.1
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

# CIFAR-10 normalization constants
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]


def normalize(image):
    """Normalize input image channel-wise to zero mean and unit variance.

    Args:
        image: Input image as float ndarray of shape (height, width, channels),
               with values in [0, 1].

    Returns:
        Normalized image with the same shape as the input.
    """
    image = image.transpose(2, 0, 1)  # Switch to channel-first
    mean, std = np.array(MEAN), np.array(STD)
    image = (image - mean[:, None, None]) / std[:, None, None]
    return image.transpose(1, 2, 0)


def apply_op(image, op, severity):
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
    pil_img = op(pil_img, severity)
    return np.asarray(pil_img) / 255.0


def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1.0):
    """Perform AugMix augmentation and compute the final mixture.

    Args:
        image: Raw input image as float32 ndarray of shape (h, w, c),
               with values in [0, 1].
        severity: Severity of augmentation operators, usually between 1 and 10.
        width: Number of augmentation chains.
        depth: Number of operations per chain. If -1, depth is sampled uniformly
               from {1, 2, 3}.
        alpha: Probability coefficient for Beta and Dirichlet distributions.

    Returns:
        Augmented and mixed normalized image.
    """
    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = np.zeros_like(image, dtype=np.float32)

    for i in range(width):
        image_aug = image.copy()
        d = depth if depth > 0 else np.random.randint(1, 4)

        for _ in range(d):
            op = np.random.choice(augmentations)
            image_aug = apply_op(image_aug, op, severity)

        # Preprocessing commutes because all coefficients are convex.
        mix += ws[i] * normalize(image_aug)

    mixed = (1 - m) * normalize(image) + m * mix
    return mixed


# ============================================================
# 3. Self-test
# ============================================================

def _self_test():
    """Run a minimal test to confirm that AugMix works."""
    np.random.seed(42)

    # Create a random CIFAR-10-like image in [0, 1].
    image = np.random.rand(IMAGE_SIZE, IMAGE_SIZE, 3).astype(np.float32)

    mixed = augment_and_mix(
        image,
        severity=3,
        width=3,
        depth=-1,
        alpha=1.0,
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
