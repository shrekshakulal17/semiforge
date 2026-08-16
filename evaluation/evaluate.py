
import os
import sys
import torch
import numpy as np

from torch.utils.data import DataLoader, random_split
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)


# --------------------------------------------------
# Imports from project
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

BATCH_SIZE = 1

# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

# --------------------------------------------------
# Create fixed validation split
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
print("Validation images:", len(val_dataset))


val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = RestorationModel()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# best_model.pth contains only model_state_dict
model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

total_psnr = 0.0
total_ssim = 0.0

num_images = 0


with torch.no_grad():

    for index, (bad, good) in enumerate(val_loader):

        bad = bad.to(device)
        good = good.to(device)

        # Model prediction
        prediction = model(bad)

        # Keep valid image range
        prediction = torch.clamp(
            prediction,
            0.0,
            1.0
        )

        # Convert tensors to numpy
        pred_np = (
            prediction[0]
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )

        good_np = (
            good[0]
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )

        # PSNR
        psnr = peak_signal_noise_ratio(
            good_np,
            pred_np,
            data_range=1.0
        )

        # SSIM
        ssim = structural_similarity(
            good_np,
            pred_np,
            channel_axis=2,
            data_range=1.0
        )

        total_psnr += psnr
        total_ssim += ssim

        num_images += 1

        if index % 10 == 0:

            print(
                f"Image [{index + 1}/{len(val_loader)}] "
                f"PSNR: {psnr:.4f} "
                f"SSIM: {ssim:.4f}"
            )


# --------------------------------------------------
# Final results
# --------------------------------------------------

average_psnr = total_psnr / num_images
average_ssim = total_ssim / num_images


print()
print("=" * 50)
print("EVALUATION COMPLETE")
print("=" * 50)

print(
    f"Average PSNR : {average_psnr:.4f} dB"
)

print(
    f"Average SSIM : {average_ssim:.4f}"
)

print(
    f"Images tested: {num_images}"
)

print("=" * 50)