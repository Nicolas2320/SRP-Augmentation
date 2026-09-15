import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet50_cifar(num_classes: int = 100) -> nn.Module:
    """
    Build an ImageNet-pretrained ResNet50 fine-tuned for CIFAR-style datasets.

    The architecture is left unmodified (original 7x7 stride-2 stem and
    maxpool) so the pretrained weights stay valid. Inputs must be resized to
    224x224 and normalized with ImageNet statistics (see get_transforms in
    train.py) to match what those weights expect. Only the final classifier
    is replaced for num_classes.

    This works for:
    - CIFAR-10 with num_classes=10
    - CIFAR-100 with num_classes=100
    """

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model