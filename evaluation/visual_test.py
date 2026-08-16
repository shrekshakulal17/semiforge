import os
import sys
import torch
import numpy as np

from PIL import Image
from torchvision.utils import save_image
from torch.utils.data import DataLoader, random_split

# --------------------------------------------------
# Add project root
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)


# --------------------------------------------------
# Imports
# --------------------------------------------------

from evaluation.evaluation_dataset import EvaluationDataset
from models.restoration import RestorationModel


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "DIV2K_train_HR"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "best_model.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "evaluation"
)

NUM_IMAGES = 5


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

# --------------------------------------------------
# Fixed validation dataset
# --------------------------------------------------

all_files = [
    f
    for f in os.listdir(DATASET_PATH)
    if f.lower().endswith(
        (".png", ".jpg", ".jpeg")
    )
]

all_files.sort()

total_images = len(all_files)

train_size = int(
    0.9 * total_images
)

val_indices = list(
    range(
        train_size,
        total_images
    )
)

val_dataset = EvaluationDataset(
    DATASET_PATH,
    val_indices,
    seed=42
)

print(
    "Total images:",
    total_images
)

print(
    "Validation images:",
    len(val_dataset)
)
# --------------------------------------------------
# Model
# --------------------------------------------------

model = RestorationModel()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Model loaded.")


# --------------------------------------------------
# Generate restored images
# --------------------------------------------------

with torch.no_grad():

    for index in range(
        min(NUM_IMAGES, len(val_dataset))
    ):

        bad, good = val_dataset[index]

        # Add batch dimension
        bad_input = bad.unsqueeze(0).to(device)

        # Model prediction
        prediction = model(bad_input)

        prediction = torch.clamp(
            prediction,
            0.0,
            1.0
        )

        # Save input
        input_path = os.path.join(
            OUTPUT_DIR,
            f"{index + 1:02d}_input.png"
        )

        save_image(
            bad,
            input_path
        )

        # Save ground truth
        ground_truth_path = os.path.join(
            OUTPUT_DIR,
            f"{index + 1:02d}_ground_truth.png"
        )

        save_image(
            good,
            ground_truth_path
        )

        # Save restored image
        restored_path = os.path.join(
            OUTPUT_DIR,
            f"{index + 1:02d}_restored.png"
        )

        save_image(
            prediction[0],
            restored_path
        )

        print(
            f"Saved test image {index + 1}"
        )


print()
print("Visual evaluation complete.")
print("Results saved to:")
print(OUTPUT_DIR)