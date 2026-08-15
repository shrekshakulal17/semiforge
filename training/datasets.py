import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


class RestorationDataset(Dataset):

    def __init__(self, image_dir, crop_size=512, noise=True):
        self.image_dir = image_dir
        self.crop_size = crop_size
        self.noise = noise

        self.images = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        self.images.sort()

    def __len__(self):
        return len(self.images)

    def add_noise(self, image):

        # Gaussian noise
        gaussian = torch.randn_like(image) * 0.03
        image = image + gaussian

        # Speckle noise
        speckle = torch.randn_like(image) * 0.08
        image = image + image * speckle

        return torch.clamp(image, 0.0, 1.0)

    def __getitem__(self, index):

        image = Image.open(self.images[index]).convert("RGB")

        width, height = image.size

        # Make sure the image is large enough
        if width < self.crop_size or height < self.crop_size:
            image = image.resize(
                (self.crop_size, self.crop_size),
                Image.Resampling.BICUBIC
            )

        else:
            # Random 512x512 crop
            left = random.randint(0, width - self.crop_size)
            top = random.randint(0, height - self.crop_size)

            image = image.crop(
                (
                    left,
                    top,
                    left + self.crop_size,
                    top + self.crop_size
                )
            )

        # Ground truth: 512x512
        ground_truth = TF.to_tensor(image)

        # Create low-resolution version: 256x256
        degraded = TF.resize(
            ground_truth,
            [256, 256],
            antialias=True
        )

        # Add synthetic noise
        if self.noise:
            degraded = self.add_noise(degraded)

        return degraded, ground_truth