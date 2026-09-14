import torch.nn as nn
from torchvision.models import resnet18

from src.config import NUM_CLASSES


def create_model():
    """
    Create the six-class SteelGuard defect classifier.

    The architecture must match the architecture
    used to train best_model.pth.
    """

    model = resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES,
    )

    return model