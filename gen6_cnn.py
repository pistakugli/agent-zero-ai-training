#!/usr/bin/env python3
"""Gen6 — CNN on synthetic image data.
Agent saw MLP maxed out on 2D points → switched to Conv2d for spatial data.
Synthetic 28x28 "images": class 0 = horizontal stripes, class 1 = vertical stripes.
"""
import torch
import torch.nn as nn
import numpy as np
import time

np.random.seed(42)

def generate_data(n):
    """Synthetic images: horizontal vs vertical stripes"""
    X = np.zeros((n, 1, 28, 28), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if i % 2 == 0:
            # Horizontal stripes
            for row in range(0, 28, 4):
                X[i, 0, row:row+2, :] = 1.0
            y[i] = 0
        else:
            # Vertical stripes
            for col in range(0, 28, 4):
                X[i, 0, :, col:col+2] = 1.0
            y[i] = 1
        # Add noise
        X[i] += np.random.randn(1, 28, 28).astype(np.float32) * 0.1
    return X, y

X, y = generate_data(1000)
X_train = torch.tensor(X[:800])
y_train = torch.tensor(y[:800])
X_val   = torch.tensor(X[800:])
y_val   = torch.tensor(y[800:])

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 28→14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 14→7
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

start = time.time()
for epoch in range(50):
    model.train()
    out = model(X_train)
    loss = criterion(out, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

elapsed = time.time() - start

model.eval()
with torch.no_grad():
    val_out = model(X_val)
    val_pred = val_out.argmax(dim=1)
    val_acc = (val_pred == y_val).float().mean().item()

print(f"✓ Gen6 CNN [Conv2d→Pool→Conv2d→Pool→FC]")
print(f"  Loss: {loss.item():.4f} | Val Accuracy: {val_acc:.4f} | Time: {elapsed:.3f}s")
print(f"  Params: {sum(p.numel() for p in model.parameters())}")
print(f"  Dataset: synthetic 28x28 stripes")
