import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.restoration import RestorationModel

# Create model
model = RestorationModel()

# Fake 256x256 RGB image
x = torch.randn(1, 3, 256, 256)

# Run model
with torch.no_grad():
    output = model(x)

print("Input shape :", x.shape)
print("Output shape:", output.shape)

print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)