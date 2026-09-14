import json
from pathlib import Path

import torch
from PIL import Image

from src.config import (
    CONFIDENCE_THRESHOLD,
    DOMAIN_MODEL_PATH,
    METADATA_PATH,
    MODEL_PATH,
)

from src.domain_gate import SteelDomainGate
from src.model import create_model
from src.preprocessing import inference_transform


# DEVICE

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# METADATA

def load_metadata():
    """
    Load metadata saved during model development.
    """

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Model metadata not found: "
            f"{METADATA_PATH}"
        )

    with open(
        METADATA_PATH,
        "r"
    ) as file:

        return json.load(file)


# DEFECT CLASSIFIER

def load_model():
    """
    Create the six-class defect classifier
    and load its trained weights.
    """

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Defect model not found: "
            f"{MODEL_PATH}"
        )

    model = create_model()

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(
        state_dict
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


# LOAD PRODUCTION ARTIFACTS ONCE

metadata = load_metadata()

model = load_model()

CLASS_NAMES = metadata[
    "class_names"
]


DOMAIN_GATE = SteelDomainGate(
    model_path=DOMAIN_MODEL_PATH,
    device=DEVICE,
)


# IMAGE PREPARATION

def _prepare_image(image):
    """
    Accept either:

        - PIL.Image.Image
        - filesystem path

    and return an RGB PIL image.
    """

    if isinstance(
        image,
        (str, Path)
    ):

        with Image.open(image) as opened_image:

            return opened_image.convert(
                "RGB"
            )

    if isinstance(
        image,
        Image.Image
    ):

        return image.convert(
            "RGB"
        )

    raise TypeError(
        "Image must be a file path "
        "or PIL Image."
    )


# PRODUCTION PREDICTION

def predict_image(image):
    """
    Complete SteelGuard inference pipeline.

    Stage 1:
        Determine whether the image belongs
        to the steel-surface domain.

    Stage 2:
        Classify valid steel images into one
        of the six supported defect classes.
    """

    # PREPARE IMAGE

    image = _prepare_image(
        image
    )


    # STAGE 1 — STEEL DOMAIN GATE

    domain_result = DOMAIN_GATE.predict(
        image
    )


    # Reject non-steel input

    if not domain_result["is_steel"]:

        return {

            "is_valid_domain":
                False,

            "predicted_class":
                None,

            "confidence":
                None,

            "status":
                "UNSUPPORTED IMAGE",

            "probabilities":
                {},

            "steel_probability":
                domain_result[
                    "steel_probability"
                ],

            "domain_threshold":
                domain_result[
                    "threshold"
                ],
        }


    # STAGE 2 — DEFECT CLASSIFICATION

    image_tensor = inference_transform(
        image
    )

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(DEVICE)
    )


    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_index = (
            torch.max(
                probabilities,
                dim=1
            )
        )


    # EXTRACT RESULT

    predicted_index = (
        predicted_index.item()
    )

    confidence = float(
        confidence.item()
    )

    predicted_class = (
        CLASS_NAMES[
            predicted_index
        ]
    )


    # PROBABILITY BREAKDOWN

    class_probabilities = {

        CLASS_NAMES[i]:
            float(
                probabilities[
                    0,
                    i
                ].item()
            )

        for i in range(
            len(CLASS_NAMES)
        )
    }


    # CONFIDENCE DECISION

    if (
        confidence
        >= CONFIDENCE_THRESHOLD
    ):

        status = (
            "DEFECT CLASSIFIED"
        )

    else:

        status = (
            "LOW CONFIDENCE — "
            "MANUAL INSPECTION"
        )


    # RETURN

    return {

        "is_valid_domain":
            True,

        "predicted_class":
            predicted_class,

        "confidence":
            confidence,

        "status":
            status,

        "probabilities":
            class_probabilities,

        "steel_probability":
            domain_result[
                "steel_probability"
            ],

        "domain_threshold":
            domain_result[
                "threshold"
            ],
    }