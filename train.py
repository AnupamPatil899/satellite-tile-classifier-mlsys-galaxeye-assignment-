import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

CLASSES = ["AnnualCrop", "Forest", "Highway", "Industrial", "Residential", "River", "SeaLake"]

class SatelliteCNN(nn.Module):
    """
    Lightweight, fast CNN designed for 64x64 satellite tile classification.
    Optimized for low-latency CPU execution in air-gapped / edge environments.
    """
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train():
    data_dir = os.path.join(
        os.path.dirname(__file__),
        "Galaxeye-BE_MLSys-TakeHome_Assignment-Tiles",
        "be-mlsys-assignment-dataset",
        "candidate_tiles"
    )

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Candidate tiles directory not found at: {data_dir}")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    classes = full_dataset.classes
    print(f"Loaded dataset with {len(full_dataset)} images across classes: {classes}")

    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    device = torch.device("cpu")
    model = SatelliteCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    epochs = 15
    print(f"Training on {device} for {epochs} epochs...")
    start_time = time.time()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch [{epoch:02d}/{epochs}] Loss: {total_loss/total:.4f} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f}s. Best Val Acc: {best_acc:.3f}")

    # Save model and metadata
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "classifier.pt")
    torch.save(model.state_dict(), model_path)
    
    metadata = {
        "classes": classes,
        "num_classes": len(classes),
        "best_val_acc": round(best_acc, 4),
        "input_size": [3, 64, 64],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "model_version": "v1.0.0-cpu",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join("models", "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model to {model_path} and metadata to models/model_metadata.json")

if __name__ == "__main__":
    train()
