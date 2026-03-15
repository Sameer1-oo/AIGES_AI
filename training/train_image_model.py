"""
AEGIS AI — Fast Image Model Training
Model: MobileNetV2 (CPU optimized, ~15 min)
"""

import os, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim import Adam

# ── Config ──────────────────────────────────────
DATASET_DIR  = "training/dataset"
WEIGHTS_PATH = "training/weights/image_model.pth"
EPOCHS       = 5        # sirf 5 epochs
BATCH_SIZE   = 16       # CPU ke liye chhota batch
LR           = 0.0005
DEVICE       = torch.device("cpu")

print("⚡ AEGIS Fast Trainer — CPU Mode")
print(f"Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")

# ── Transforms ──────────────────────────────────
tfm = transforms.Compose([
    transforms.Resize((128, 128)),   # 224 nahi, 128 — 3x fast
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# ── Data ────────────────────────────────────────
train_data   = datasets.ImageFolder(os.path.join(DATASET_DIR,"Train"), transform=tfm)
test_data    = datasets.ImageFolder(os.path.join(DATASET_DIR,"Test"),  transform=tfm)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Classes : {train_data.classes}")
print(f"Train   : {len(train_data)} images")
print(f"Test    : {len(test_data)} images")

# ── Model: MobileNetV2 ──────────────────────────
model = models.mobilenet_v2(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(1280, 2)   # Fake / Real
model = model.to(DEVICE)

optimizer = Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ── Train ────────────────────────────────────────
best_acc = 0.0
os.makedirs("training/weights", exist_ok=True)

for epoch in range(EPOCHS):
    # Train
    model.train()
    correct, total = 0, 0
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        correct += (out.argmax(1) == labels).sum().item()
        total   += labels.size(0)

    train_acc = correct / total * 100

    # Eval
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            out      = model(imgs)
            correct += (out.argmax(1) == labels).sum().item()
            total   += labels.size(0)

    test_acc = correct / total * 100
    print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_acc:.1f}% | Test: {test_acc:.1f}%")

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), WEIGHTS_PATH)
        print(f"  ✅ Saved! Best: {best_acc:.1f}%")

print(f"\n🏁 Done! Best Accuracy: {best_acc:.1f}%")
print(f"Model: {WEIGHTS_PATH}")