#!/usr/bin/env python3
"""Gen2 — MLP [64→32→10]. Agent saw linear underfits → added hidden layer.
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

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
    def forward(self, x):
        return self.net(x)

model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

start = time.time()
for epoch in range(200):
    out = model(X_train_t)
    loss = criterion(out, y_train_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
elapsed = time.time() - start

model.eval()
with torch.no_grad():
    val_acc = (model(X_val_t).argmax(1) == y_val_t).float().mean().item()

print(f"✓ Gen2 MLP [64→32→10]")
print(f"  Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f} | Time: {elapsed:.2f}s")
print(f"  Params: {sum(p.numel() for p in model.parameters())}")
