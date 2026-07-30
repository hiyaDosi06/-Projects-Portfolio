import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

# 1. Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Fetch LFW Dataset (Color = True for 3-channel input)
print("Loading LFW dataset...")
lfw_color = fetch_lfw_people(min_faces_per_person=70, color=True, resize=0.5)
X, y = lfw_color.images, lfw_color.target
target_names = lfw_color.target_names
n_classes = len(target_names)

# Convert shape to PyTorch format (N, C, H, W) and scale pixels to [0, 1]
X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
y = torch.tensor(y, dtype=torch.long)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

train_loader = DataLoader(
    TensorDataset(X_train, y_train), batch_size=32, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(X_test, y_test), batch_size=32, shuffle=False
)

# 3. Fine-Tune Pre-trained ResNet-18
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Replace final fully connected layer to match number of LFW classes
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, n_classes)
resnet = resnet.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr=0.0003)

# 4. Training Loop
num_epochs = 15
print("\nStarting Training...")
for epoch in range(num_epochs):
    resnet.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # Clean backpropagation
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}"
    )

# 5. Evaluation Loop
resnet.eval()
all_preds, all_targets = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = resnet(inputs)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

print("\n" + "=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)
print(
    classification_report(all_targets, all_preds, target_names=target_names)
)