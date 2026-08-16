'''import os
import sys
import time

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import torch
from torch.utils.data import DataLoader, random_split

from dataset import RestorationDataset
from models.restoration import RestorationModel
from losses import RestorationLoss


# =========================
# CONFIG
# =========================

DATASET_PATH = "datasets/DIV2K_train_HR"

BATCH_SIZE = 1
EPOCHS = 5

LEARNING_RATE = 2e-4

NUM_WORKERS = 0

CHECKPOINT_DIR = "outputs/checkpoints"


# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# =========================
# DATASET
# =========================

dataset = RestorationDataset(DATASET_PATH)

print("Total images:", len(dataset))


# 90% training / 10% validation

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# =========================
# MODEL
# =========================

model = RestorationModel().to(device)

print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)


# =========================
# LOSS
# =========================

criterion = RestorationLoss()


# =========================
# OPTIMIZER
# =========================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# =========================
# SCHEDULER
# =========================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# =========================
# CHECKPOINT DIRECTORY
# =========================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# =========================
# BEST MODEL
# =========================

best_val_loss = float("inf")


# =========================
# TRAINING
# =========================

for epoch in range(EPOCHS):

    # -------------------------
    # TRAIN
    # -------------------------

    model.train()

    running_train_loss = 0.0

    start_time = time.time()

    for batch_index, (bad, good) in enumerate(train_loader):

        bad = bad.to(device)
        good = good.to(device)

        # Forward
        prediction = model(bad)

        # Loss
        loss = criterion(
            prediction,
            good
        )

        # Clear gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_train_loss += loss.item()

        if batch_index % 50 == 0:

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{batch_index}/{len(train_loader)}] "
                f"Loss: {loss.item():.6f}"
            )

    average_train_loss = (
        running_train_loss / len(train_loader)
    )


    # -------------------------
    # VALIDATION
    # -------------------------

    model.eval()

    running_val_loss = 0.0

    with torch.no_grad():

        for bad, good in val_loader:

            bad = bad.to(device)
            good = good.to(device)

            prediction = model(bad)

            loss = criterion(
                prediction,
                good
            )

            running_val_loss += loss.item()

    average_val_loss = (
        running_val_loss / len(val_loader)
    )


    # -------------------------
    # SCHEDULER
    # -------------------------

    scheduler.step()


    # -------------------------
    # TIME
    # -------------------------

    epoch_time = time.time() - start_time


    print()
    print("=" * 50)
    print(f"Epoch {epoch + 1}/{EPOCHS} finished")
    print(f"Train Loss: {average_train_loss:.6f}")
    print(f"Val Loss:   {average_val_loss:.6f}")
    print(f"Time:       {epoch_time:.2f} seconds")
    print("=" * 50)


    # -------------------------
    # SAVE EPOCH CHECKPOINT
    # -------------------------

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"model_epoch_{epoch + 1}.pth"
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": average_train_loss,
            "val_loss": average_val_loss,
        },
        checkpoint_path
    )

    print("Saved:", checkpoint_path)


    # -------------------------
    # SAVE BEST MODEL
    # -------------------------

    if average_val_loss < best_val_loss:

        best_val_loss = average_val_loss

        torch.save(
            model.state_dict(),
            "outputs/best_model.pth"
        )

        print(
            "⭐ New best model saved!"
        )


print()
print("Training complete.")
print(
    f"Best validation loss: {best_val_loss:.6f}"
)'''
import os
import sys
import time

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import torch
from torch.utils.data import DataLoader, random_split

from dataset import RestorationDataset
from models.restoration import RestorationModel
from losses import RestorationLoss


# =========================
# CONFIG
# =========================

DATASET_PATH = "datasets/DIV2K_train_HR"

BATCH_SIZE = 1
EPOCHS = 5

LEARNING_RATE = 2e-4
NUM_WORKERS = 0

CHECKPOINT_DIR = "outputs/checkpoints"

RESUME_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "model_epoch_4.pth"
)


# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# =========================
# DATASET
# =========================

dataset = RestorationDataset(DATASET_PATH)

print("Total images:", len(dataset))

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# =========================
# MODEL
# =========================

model = RestorationModel().to(device)

print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)


# =========================
# LOSS
# =========================

criterion = RestorationLoss()


# =========================
# OPTIMIZER
# =========================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# =========================
# LOAD EPOCH 4
# =========================

print()
print("Loading checkpoint:")
print(RESUME_CHECKPOINT)

checkpoint = torch.load(
    RESUME_CHECKPOINT,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)

start_epoch = checkpoint["epoch"]

best_val_loss = checkpoint["val_loss"]

print("Resuming after epoch:", start_epoch)
print("Best validation loss:", best_val_loss)


# =========================
# TRAIN ONLY EPOCH 5
# =========================

epoch = start_epoch

model.train()

running_train_loss = 0.0

start_time = time.time()

for batch_index, (bad, good) in enumerate(train_loader):

    bad = bad.to(device)
    good = good.to(device)

    prediction = model(bad)

    loss = criterion(
        prediction,
        good
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    running_train_loss += loss.item()

    if batch_index % 50 == 0:

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Batch [{batch_index}/{len(train_loader)}] "
            f"Loss: {loss.item():.6f}"
        )


average_train_loss = (
    running_train_loss / len(train_loader)
)


# =========================
# VALIDATION
# =========================

model.eval()

running_val_loss = 0.0

with torch.no_grad():

    for bad, good in val_loader:

        bad = bad.to(device)
        good = good.to(device)

        prediction = model(bad)

        loss = criterion(
            prediction,
            good
        )

        running_val_loss += loss.item()


average_val_loss = (
    running_val_loss / len(val_loader)
)


epoch_time = time.time() - start_time


# =========================
# RESULTS
# =========================

print()
print("=" * 50)
print("Epoch 5 finished")
print(f"Train Loss: {average_train_loss:.6f}")
print(f"Val Loss:   {average_val_loss:.6f}")
print(f"Time:       {epoch_time:.2f} seconds")
print("=" * 50)


# =========================
# SAVE EPOCH 5
# =========================

checkpoint_path = os.path.join(
    CHECKPOINT_DIR,
    "model_epoch_5.pth"
)

torch.save(
    {
        "epoch": 5,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": average_train_loss,
        "val_loss": average_val_loss,
    },
    checkpoint_path
)

print("Saved:", checkpoint_path)


# =========================
# SAVE BEST MODEL
# =========================

if average_val_loss < best_val_loss:

    torch.save(
        model.state_dict(),
        "outputs/best_model.pth"
    )

    print("⭐ Epoch 5 is the NEW best model.")

else:

    print("Epoch 4 remains the best model.")


print()
print("Training complete.")