import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Learns image corrections while preserving useful information.
    """

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class RestorationModel(nn.Module):
    """
    Lightweight image restoration + 2x super-resolution model.

    Input:
        [B, 3, 256, 256]

    Output:
        [B, 3, 512, 512]
    """

    def __init__(self, channels=48, num_blocks=6):
        super().__init__()

        # Initial feature extraction
        self.head = nn.Conv2d(
            3,
            channels,
            kernel_size=3,
            padding=1
        )

        # Main restoration body
        self.body = nn.Sequential(
            *[
                ResidualBlock(channels)
                for _ in range(num_blocks)
            ]
        )

        # Feature refinement
        self.body_tail = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        # 2x upscaling
        self.upsample = nn.Sequential(
            nn.Conv2d(
                channels,
                channels * 4,
                kernel_size=3,
                padding=1
            ),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        # Final image reconstruction
        self.output = nn.Conv2d(
            channels,
            3,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        # Bicubic baseline.
        # The network learns corrections on top of this.
        base = F.interpolate(
            x,
            scale_factor=2,
            mode="bicubic",
            align_corners=False
        )

        # Extract features
        features = self.head(x)

        # Residual restoration
        residual = self.body(features)
        residual = self.body_tail(residual)

        # Global residual connection
        residual = residual + features

        # Upscale features
        residual = self.upsample(residual)

        # Convert features into RGB correction
        residual = self.output(residual)

        # Add learned correction to bicubic result
        output = base + residual

        # Keep image values valid
        output = torch.clamp(output, 0.0, 1.0)

        return output