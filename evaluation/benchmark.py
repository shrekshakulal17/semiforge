import os
import sys
import time
import torch

# --------------------------------------------------
# Project root
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)


from models.restoration import RestorationModel


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "best_model.pth"
)

NUM_RUNS = 20


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Load model
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
# Test image
# --------------------------------------------------

x = torch.randn(
    1,
    3,
    256,
    256
).to(device)


# --------------------------------------------------
# Warm-up
# --------------------------------------------------

print("Warming up...")

with torch.no_grad():

    for _ in range(5):
        _ = model(x)


if device.type == "cuda":
    torch.cuda.synchronize()


# --------------------------------------------------
# Benchmark
# --------------------------------------------------

times = []

print("Running benchmark...")


with torch.no_grad():

    for i in range(NUM_RUNS):

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        elapsed_ms = (
            end - start
        ) * 1000

        times.append(elapsed_ms)

        print(
            f"Run {i + 1:02d}: "
            f"{elapsed_ms:.2f} ms"
        )


# --------------------------------------------------
# Results
# --------------------------------------------------

average_time = sum(times) / len(times)

minimum_time = min(times)
maximum_time = max(times)


print()
print("=" * 50)
print("INFERENCE BENCHMARK")
print("=" * 50)

print(
    f"Average: {average_time:.2f} ms"
)

print(
    f"Minimum: {minimum_time:.2f} ms"
)

print(
    f"Maximum: {maximum_time:.2f} ms"
)

print(
    f"Approx FPS: {1000 / average_time:.2f}"
)

print("=" * 50)