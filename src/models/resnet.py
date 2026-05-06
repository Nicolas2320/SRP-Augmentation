import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18_cifar10(num_classes: int = 10) -> nn.Module:
    """
    Build ResNet18 adapted for CIFAR-10.

    Standard ResNet18 was designed for larger ImageNet images.
    CIFAR-10 images are 32x32, so we use:
    - 3x3 first convolution
    - stride 1
    - no initial maxpool
    - output layer with 10 classes
    """
    model = resnet18(weights=None, num_classes=num_classes)

    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    model.maxpool = nn.Identity()

    return model