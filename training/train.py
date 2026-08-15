import os
import time
import sys


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
EPOCHS = 1

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

full_dataset = RestorationDataset(DATASET_PATH)

dataset, _ = random_split(
    full_dataset,
    [20, len(full_dataset) - 20],
    generator=torch.Generator().manual_seed(42)
)

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

model = RestorationModel()

model = model.to(device)

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
# TRAINING
# =========================

best_loss = float("inf")


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    start_time = time.time()

    for batch_index, (bad, good) in enumerate(train_loader):

        bad = bad.to(device)
        good = good.to(device)

        # Forward pass
        prediction = model(bad)

        # Calculate loss
        loss = criterion(
            prediction,
            good
        )

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

        # Print progress
        if batch_index % 10 == 0:

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{batch_index}/{len(train_loader)}] "
                f"Loss: {loss.item():.6f}"
            )

    scheduler.step()

    average_loss = (
        running_loss / len(train_loader)
    )

    epoch_time = time.time() - start_time

    print()
    print(
        f"Epoch {epoch + 1} finished"
    )

    print(
        f"Average Loss: {average_loss:.6f}"
    )

    print(
        f"Time: {epoch_time:.2f} seconds"
    )


    # =========================
    # SAVE CHECKPOINT
    # =========================

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"model_epoch_{epoch + 1}.pth"
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": average_loss,
        },
        checkpoint_path
    )

    print(
        "Saved:",
        checkpoint_path
    )


    # Save best model

    if average_loss < best_loss:

        best_loss = average_loss

        torch.save(
            model.state_dict(),
            "outputs/best_model.pth"
        )

        print(
            "New best model saved."
        )

print()
print("Training complete.")