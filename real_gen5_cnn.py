#!/usr/bin/env python3
"""Gen5 — CNN on 8x8 digit images.
Agent saw MLP treats pixels as flat vector → reshape to 1x8x8, use Conv2d.
Spatial structure matters for digits.
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

# Reshape for CNN: (N, 64) → (N, 1, 8, 8)
X_train_img = X_train_t.view(-1, 1, 8, 8)
X_val_img   = X_val_t.view(-1, 1, 8, 8)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 8x8→8x8
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 8x8→4x4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 4x4→4x4
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 4x4→2x2
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

start = time.time()
for epoch in range(200):
    model.train()
    out = model(X_train_img)
    loss = criterion(out, y_train_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
elapsed = time.time() - start

model.eval()
with torch.no_grad():
    val_acc = (model(X_val_img).argmax(1) == y_val_t).float().mean().item()

print(f"✓ Gen5 CNN [Conv32→Pool→Conv64→Pool→FC]")
print(f"  Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f} | Time: {elapsed:.2f}s")
print(f"  Params: {sum(p.numel() for p in model.parameters())}")
