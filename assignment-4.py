import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ==========================================
# 1. Configuration & Setup
# ==========================================
DATA_DIR = "./dataset"  # Expects ./dataset/train and ./dataset/val
BATCH_SIZE = 16
NUM_CLASSES = 1  # 1 for Binary (Tumor vs No Tumor)
NUM_EPOCHS = 5
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ==========================================
# 2. Data Transformations
# ==========================================
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    ),
}


# ==========================================
# 3. Model Architecture (ResNet-50)
# ==========================================
def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze earlier layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4 for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Custom FC Classification Head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, NUM_CLASSES),
    )
    return model.to(DEVICE)


# ==========================================
# 4. Training Loop
# ==========================================
def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE).float().unsqueeze(1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(
                dataloaders[phase].dataset
            )

            print(
                f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}"
            )

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc

    print(f"\nTraining Complete. Best Validation Accuracy: {best_acc:.4f}")
    return model


# ==========================================
# 5. Main Execution Entry Point
# ==========================================
if __name__ == "__main__":
    print("Initializing ResNet-50 Model for MRI Cancer Detection...")
    model = build_model()
    print("Model initialized successfully!")

    # Check if dataset path exists to run training
    if os.path.exists(os.path.join(DATA_DIR, "train")):
        image_datasets = {
            x: datasets.ImageFolder(
                os.path.join(DATA_DIR, x), data_transforms[x]
            )
            for x in ["train", "val"]
        }

        dataloaders = {
            x: DataLoader(
                image_datasets[x],
                batch_size=BATCH_SIZE,
                shuffle=(x == "train"),
            )
            for x in ["train", "val"]
        }

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE,
        )

        train_model(
            model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS
        )
    else:
        print(f"\n[INFO] Model structure ready.")
        print(
            f"To train, add dataset images into: {DATA_DIR}/train/ and {DATA_DIR}/val/"
        )