import os
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


class EvaluationDataset(Dataset):

    def __init__(
        self,
        image_dir,
        indices,
        crop_size=512,
        seed=42
    ):
        self.image_dir = image_dir
        self.crop_size = crop_size

        self.images = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(
                (".png", ".jpg", ".jpeg")
            )
        ]

        self.images.sort()

        # Use only the selected validation images
        self.images = [
            self.images[i]
            for i in indices
        ]

        self.seed = seed


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):

        # Fixed random generator for reproducibility
        rng = random.Random(
            self.seed + index
        )

        image = Image.open(
            self.images[index]
        ).convert("RGB")

        width, height = image.size


        # --------------------------------------------------
        # Create deterministic 512x512 crop
        # --------------------------------------------------

        if (
            width < self.crop_size
            or height < self.crop_size
        ):

            image = image.resize(
                (
                    self.crop_size,
                    self.crop_size
                ),
                Image.Resampling.BICUBIC
            )

        else:

            left = rng.randint(
                0,
                width - self.crop_size
            )

            top = rng.randint(
                0,
                height - self.crop_size
            )

            image = image.crop(
                (
                    left,
                    top,
                    left + self.crop_size,
                    top + self.crop_size
                )
            )


        # --------------------------------------------------
        # Ground truth
        # --------------------------------------------------

        ground_truth = TF.to_tensor(
            image
        )


        # --------------------------------------------------
        # 512 → 256
        # --------------------------------------------------

        degraded = TF.resize(
            ground_truth,
            [256, 256],
            antialias=True
        )


        # --------------------------------------------------
        # Gaussian noise
        # --------------------------------------------------

        noise_generator = torch.Generator()

        noise_generator.manual_seed(
            self.seed + index
        )

        gaussian = (
            torch.randn(
                degraded.shape,
                generator=noise_generator
            ) * 0.03
        )

        degraded = (
            degraded + gaussian
        )


        # --------------------------------------------------
        # Speckle noise
        # --------------------------------------------------

        speckle_generator = torch.Generator()

        speckle_generator.manual_seed(
            self.seed + index + 10000
        )

        speckle = (
            torch.randn(
                degraded.shape,
                generator=speckle_generator
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