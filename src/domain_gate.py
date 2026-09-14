from pathlib import Path

import torch
import torch.nn as nn

from PIL import Image

from torchvision import models, transforms


class SteelDomainGate:
    """
    Stage 1 of SteelGuard.

    Determines whether an uploaded image belongs
    to the steel-surface inspection domain.

    Class mapping:

        0 -> non_steel
        1 -> steel
    """

    def __init__(
        self,
        model_path,
        device,
    ):
        self.device = device

        self.model_path = Path(
            model_path
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Steel domain model not found: "
                f"{self.model_path}"
            )

        
        # MUST MATCH DOMAIN MODEL VALIDATION PREPROCESSING

        self.transform = transforms.Compose([
            transforms.Grayscale(
                num_output_channels=3
            ),

            transforms.Resize(
                (224, 224)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # LOAD CHECKPOINT

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,
        )

        self.threshold = float(
            checkpoint["threshold"]
        )

        # BUILD DOMAIN MODEL

        self.model = models.resnet18(
            weights=None
        )

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            2,
        )

        # LOAD TRAINED WEIGHTS

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

    def predict(self, image):
        """
        Determine whether the image belongs
        to the steel domain.
        """

        # ACCEPT PIL IMAGE OR FILE PATH

        if isinstance(
            image,
            (str, Path)
        ):

            with Image.open(image) as opened_image:

                image = opened_image.convert(
                    "RGB"
                )

        elif isinstance(
            image,
            Image.Image
        ):

            image = image.convert(
                "RGB"
            )

        else:

            raise TypeError(
                "Image must be a file path "
                "or PIL Image."
            )

        # PREPROCESS

        image_tensor = self.transform(
            image
        )

        image_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(self.device)
        )

        # INFERENCE

        with torch.no_grad():

            logits = self.model(
                image_tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

        # STEEL PROBABILITY

        steel_probability = float(
            probabilities[0, 1].item()
        )

        is_steel = (
            steel_probability
            >= self.threshold
        )

        return {
            "is_steel": is_steel,

            "steel_probability":
                steel_probability,

            "threshold":
                self.threshold,
        }