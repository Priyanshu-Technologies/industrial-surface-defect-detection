from torchvision import transforms

from src.config import IMAGE_SIZE


# DEFECT MODEL INFERENCE PREPROCESSING

inference_transform = transforms.Compose([
    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.Resize(
        IMAGE_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])