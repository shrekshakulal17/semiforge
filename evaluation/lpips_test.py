import os
import sys
import random

import torch
import lpips

from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# MODEL
# ============================================================

from models.restoration import RestorationModel


# ============================================================
# CONFIG
# ============================================================

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
NUM_WORKERS = 0


# ============================================================
# FIXED EVALUATION DATASET
# ============================================================

class LPIPSEvaluationDataset(Dataset):

    def __init__(
        self,
        image_dir,
        seed=42
    ):

        self.image_dir = image_dir
        self.seed = seed

        self.images = [
            os.path.join(
                image_dir,
                f
            )
            for f in os.listdir(image_dir)
            if f.lower().endswith(
                (".png", ".jpg", ".jpeg")
            )
        ]

        self.images.sort()

        # Same 90/10 split used during training
        train_size = int(
            0.9 * len(self.images)
        )

        self.images = self.images[
            train_size:
        ]


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):

        rng = random.Random(
            self.seed + index
        )

        image = Image.open(
            self.images[index]
        ).convert("RGB")


        width, height = image.size

        crop_size = 512


        # ----------------------------------------------------
        # Fixed 512x512 crop
        # ----------------------------------------------------

        if (
            width < crop_size
            or height < crop_size
        ):

            image = image.resize(
                (
                    crop_size,
                    crop_size
                ),
                Image.Resampling.BICUBIC
            )

        else:

            left = rng.randint(
                0,
                width - crop_size
            )

            top = rng.randint(
                0,
                height - crop_size
            )

            image = image.crop(
                (
                    left,
                    top,
                    left + crop_size,
                    top + crop_size
                )
            )


        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        ground_truth = TF.to_tensor(
            image
        )


        # ----------------------------------------------------
        # 512 → 256
        # ----------------------------------------------------

        degraded = TF.resize(
            ground_truth,
            [256, 256],
            antialias=True
        )


        # ----------------------------------------------------
        # Gaussian noise
        # ----------------------------------------------------

        generator = torch.Generator()

        generator.manual_seed(
            self.seed + index
        )

        gaussian = (
            torch.randn(
                degraded.shape,
                generator=generator
            ) * 0.03
        )

        degraded = (
            degraded + gaussian
        )


        # ----------------------------------------------------
        # Speckle noise
        # ----------------------------------------------------

        generator2 = torch.Generator()

        generator2.manual_seed(
            self.seed + index + 10000
        )

        speckle = (
            torch.randn(
                degraded.shape,
                generator=generator2
            ) * 0.08
        )

        degraded = (
            degraded
            + degraded * speckle
        )


        degraded = torch.clamp(
            degraded,
            0.0,
            1.0
        )


        return degraded, ground_truth


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# DATASET
# ============================================================

dataset = LPIPSEvaluationDataset(
    DATASET_PATH
)

print(
    "Validation images:",
    len(dataset)
)


val_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# LOAD RESTORATION MODEL
# ============================================================

model = RestorationModel()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)


if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(device)

model.eval()

print(
    "Model loaded successfully."
)


# ============================================================
# LOAD LPIPS
# ============================================================

print(
    "Loading LPIPS..."
)

loss_fn = lpips.LPIPS(
    net="alex"
).to(device)

loss_fn.eval()

print(
    "LPIPS loaded successfully."
)


# ============================================================
# CALCULATE LPIPS
# ============================================================

total_lpips = 0.0
num_images = 0


with torch.no_grad():

    for index, (bad, good) in enumerate(
        val_loader
    ):

        bad = bad.to(device)
        good = good.to(device)


        # ----------------------------------------------------
        # Restoration
        # ----------------------------------------------------

        prediction = model(
            bad
        )

        prediction = torch.clamp(
            prediction,
            0.0,
            1.0
        )


        # ----------------------------------------------------
        # Convert [0,1] → [-1,1]
        # ----------------------------------------------------

        prediction_lpips = (
            prediction * 2.0
        ) - 1.0

        good_lpips = (
            good * 2.0
        ) - 1.0


        # ----------------------------------------------------
        # LPIPS
        # ----------------------------------------------------

        score = loss_fn(
            prediction_lpips,
            good_lpips
        )


        score_value = score.mean().item()


        total_lpips += (
            score_value
            * bad.size(0)
        )

        num_images += bad.size(0)


        if index % 10 == 0:

            print(
                f"Image [{num_images}/{len(dataset)}] "
                f"LPIPS: {score_value:.4f}"
            )


# ============================================================
# FINAL RESULT
# ============================================================

average_lpips = (
    total_lpips / num_images
)


print()
print("=" * 50)
print(
    "LPIPS EVALUATION COMPLETE"
)
print("=" * 50)

print(
    f"Average LPIPS : {average_lpips:.4f}"
)

print(
    f"Images tested : {num_images}"
)

print(
    "Lower LPIPS is better."
)

print("=" * 50)