import os
import sys
import time
import tkinter as tk
from tkinter import filedialog

import torch
from PIL import Image
import torchvision.transforms.functional as TF


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# MODEL
# ============================================================

from models.restoration import RestorationModel


# ============================================================
# SELECT IMAGE
# ============================================================

root = tk.Tk()
root.withdraw()

print("Select an image to restore...")

input_path = filedialog.askopenfilename(
    title="Select image",
    filetypes=[
        (
            "Image files",
            "*.png *.jpg *.jpeg *.bmp *.webp"
        ),
        (
            "All files",
            "*.*"
        )
    ]
)

root.destroy()


if not input_path:

    print("No image selected.")
    sys.exit(0)


print()
print("Selected image:")
print(input_path)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "best_model.pth"
)


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


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
# LOAD MODEL
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

print("Model loaded successfully.")


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(
    input_path
).convert("RGB")


original_width, original_height = image.size

print(
    f"Input resolution: "
    f"{original_width} × {original_height}"
)


# ============================================================
# PREPROCESS
# ============================================================

model_input = image.resize(
    (256, 256),
    Image.Resampling.BICUBIC
)

tensor = TF.to_tensor(
    model_input
)

tensor = tensor.unsqueeze(0)

tensor = tensor.to(device)


# ============================================================
# INFERENCE
# ============================================================

if device.type == "cuda":
    torch.cuda.synchronize()

start = time.perf_counter()


with torch.no_grad():

    prediction = model(
        tensor
    )


if device.type == "cuda":
    torch.cuda.synchronize()

end = time.perf_counter()


inference_time = (
    end - start
) * 1000


# ============================================================
# OUTPUT
# ============================================================

prediction = prediction.squeeze(
    0
).cpu()

prediction = torch.clamp(
    prediction,
    0.0,
    1.0
)

output_image = TF.to_pil_image(
    prediction
)


# ============================================================
# SAVE RESULT
# ============================================================

output_dir = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(
    output_dir,
    exist_ok=True
)


base_name = os.path.splitext(
    os.path.basename(input_path)
)[0]

output_path = os.path.join(
    output_dir,
    f"{base_name}_restored.png"
)


output_image.save(
    output_path
)


# ============================================================
# RESULTS
# ============================================================

output_width, output_height = (
    output_image.size
)


print()
print("=" * 60)
print("RESTORATION COMPLETE")
print("=" * 60)

print(
    f"Input resolution:  "
    f"{original_width} × {original_height}"
)

print(
    f"Model input:       256 × 256"
)

print(
    f"Output resolution: "
    f"{output_width} × {output_height}"
)

print(
    f"Inference time:    "
    f"{inference_time:.2f} ms"
)

print(
    f"Device:            "
    f"{device}"
)

print(
    f"Output saved to:   "
    f"{output_path}"
)

print("=" * 60)