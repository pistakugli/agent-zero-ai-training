#!/usr/bin/env python3
"""Gen6 — CNN + BatchNorm + LR scheduler + early stopping.
Production-grade training loop. Agent saw basic CNN → added full pipeline.
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import time

# REAL data — sklearn digits, 1797 handwritten 8x8 images, 10 classes
digits = load_digits()
X = digits.data.astype(np.float32) / 16.0   # normalize [0,1]
y = digits.target.astype(np.int64)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_val_t   = torch.tensor(X_val)
y_val_t   = torch.tensor(y_val)

print(f"  Dataset: sklearn digits (REAL handwritten)")
print(f"  Train: {X_train_t.shape[0]} | Val: {X_val_t.shape[0]} | Classes: 10")

X_train_img = X_train_t.view(-1, 1, 8, 8)
X_val_img   = X_val_t.view(-1, 1, 8, 8)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 8→4
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 4→2
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0
best_val_loss = float("inf")
patience = 20
no_improve = 0

start = time.time()
for epoch in range(300):
    model.train()
    out = model(X_train_img)
    loss = criterion(out, y_train_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_out = model(X_val_img)
        val_loss = criterion(val_out, y_val_t).item()
        val_acc = (val_out.argmax(1) == y_val_t).float().mean().item()

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            break

elapsed = time.time() - start

print(f"✓ Gen6 CNN + BN + Scheduler + EarlyStopping")
print(f"  Loss: {loss.item():.4f} | Best Val Acc: {best_val_acc:.4f} | Epochs: {epoch+1}")
print(f"  Time: {elapsed:.2f}s | Final LR: {optimizer.param_groups[0]['lr']:.6f}")
print(f"  Params: {sum(p.numel() for p in model.parameters())}")
