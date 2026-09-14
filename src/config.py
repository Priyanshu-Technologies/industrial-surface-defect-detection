from pathlib import Path


# PROJECT

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"


# DEFECT CLASSIFIER

MODEL_PATH = MODEL_DIR / "best_model.pth"

METADATA_PATH = MODEL_DIR / "metadata.json"


# STEEL DOMAIN GATE

DOMAIN_MODEL_PATH = (
    MODEL_DIR / "steel_domain_model.pth"
)


# INFERENCE

IMAGE_SIZE = (224, 224)

NUM_CLASSES = 6

CONFIDENCE_THRESHOLD = 0.70