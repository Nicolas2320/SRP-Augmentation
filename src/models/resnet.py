import torch.nn as nn
from torchvision.models import resnet50


def build_resnet50_cifar(num_classes: int = 100) -> nn.Module:
    """
    Build ResNet50 adapted for CIFAR-style datasets.

    Standard ResNet50 was designed for ImageNet images of size 224x224.
    CIFAR images are 32x32, so we adapt the first layers:

    - use a 3x3 first convolution instead of 7x7
    - use stride 1 instead of stride 2
    - remove the initial maxpool
    - set the final classifier to num_classes

    This works for:
    - CIFAR-10 with num_classes=10
    - CIFAR-100 with num_classes=100
    """

    model = resnet50(weights=None, num_classes=num_classes)

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