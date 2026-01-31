#!/usr/bin/env python3
"""Gen7 — Full CNN pipeline: BatchNorm + LR scheduling + early stopping.
Agent saw basic CNN → added production-grade training loop.
Harder dataset: 4 classes of synthetic patterns.
"""
import torch
import torch.nn as nn
import numpy as np
import time

np.random.seed(42)

def generate_data(n):
    """4 classes: H-stripes, V-stripes, diagonal, dots"""
    X = np.zeros((n, 1, 28, 28), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        cls = i % 4
        y[i] = cls
        if cls == 0:  # Horizontal
            for r in range(0, 28, 4): X[i, 0, r:r+2, :] = 1.0
        elif cls == 1:  # Vertical
            for c in range(0, 28, 4): X[i, 0, :, c:c+2] = 1.0
        elif cls == 2:  # Diagonal
            for d in range(28): X[i, 0, d, d] = 1.0; X[i, 0, min(d+1,27), d] = 1.0
        else:  # Dots
            for r in range(2, 28, 5):
                for c in range(2, 28, 5):
                    X[i, 0, r, c] = 1.0
        X[i] += np.random.randn(1, 28, 28).astype(np.float32) * 0.15
    return X, y

X, y = generate_data(2000)
X_train = torch.tensor(X[:1600])
y_train = torch.tensor(y[:1600])
X_val   = torch.tensor(X[1600:])
y_val   = torch.tensor(y[1600:])

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),              # 28→14
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),              # 14→7
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),      # 7→1
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 4)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0
patience = 10
no_improve = 0
best_loss = float("inf")

start = time.time()
for epoch in range(100):
    model.train()
    out = model(X_train)
    loss = criterion(out, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_out = model(X_val)
        val_loss = criterion(val_out, y_val).item()
        val_acc = (val_out.argmax(1) == y_val).float().mean().item()
    
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        best_val_acc = val_acc
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            break

elapsed = time.time() - start
final_lr = optimizer.param_groups[0]["lr"]

print(f"✓ Gen7 CNN+BatchNorm+Scheduler+EarlyStopping")
print(f"  Loss: {loss.item():.4f} | Best Val Acc: {best_val_acc:.4f} | Epochs: {epoch+1}")
print(f"  Time: {elapsed:.3f}s | Final LR: {final_lr:.6f}")
print(f"  Params: {sum(p.numel() for p in model.parameters())}")
print(f"  Dataset: 4-class synthetic 28x28")
