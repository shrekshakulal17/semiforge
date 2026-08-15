from PIL import Image
import numpy as np

# Your first image
input_path = "datasets/DIV2K_train_HR/0001x2.png"

# Open image
image = Image.open(input_path).convert("RGB")

# Take a 512 x 512 crop from the top-left
crop = image.crop((0, 0, 512, 512))

# This is our temporary "good" image
crop.save("outputs/good_512.png")

# Make it 256 x 256
bad = crop.resize((256, 256), Image.Resampling.BICUBIC)

# Convert to NumPy
bad_array = np.array(bad).astype(np.float32) / 255.0

# Gaussian noise
gaussian = np.random.normal(
    0,
    0.03,
    bad_array.shape
)

bad_array = bad_array + gaussian

# Speckle noise
speckle = np.random.normal(
    0,
    0.08,
    bad_array.shape
)

bad_array = bad_array + bad_array * speckle

# Keep values between 0 and 1
bad_array = np.clip(bad_array, 0, 1)

# Convert back to image
bad = Image.fromarray(
    (bad_array * 255).astype(np.uint8)
)

bad.save("outputs/bad_256.png")

print("Done!")
print("Good image: outputs/good_512.png")
print("Bad image:  outputs/bad_256.png")