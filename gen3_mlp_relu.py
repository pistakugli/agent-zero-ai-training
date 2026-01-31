#!/usr/bin/env python3
"""Gen3 — MLP: input → hidden(16) → ReLU → output.
First time using nn.Module. Agent saw linear was too simple → added hidden layer.
"""
import torch
import torch.nn as nn
import numpy as np
import time

np.random.seed(42)
N = 1000
X = np.random.randn(N, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)

X_train = torch.tensor(X[:800], dtype=torch.float32)
y_train = torch.tensor(y[:800], dtype=torch.float32).unsqueeze(1)
X_val   = torch.tensor(X[800:], dtype=torch.float32)
y_val   = torch.tensor(y[800:], dtype=torch.float32)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
criterion = nn.BCELoss()

start = time.time()
for epoch in range(100):
    pred = model(X_train)
    loss = criterion(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

elapsed = time.time() - start

with torch.no_grad():
    val_pred = model(X_val)
    val_acc = ((val_pred > 0.5).squeeze() == y_val).float().mean().item()

print(f"✓ Gen3 MLP [2→16→1] + ReLU")
print(f"  Loss: {loss.item():.4f} | Val Accuracy: {val_acc:.4f} | Time: {elapsed:.3f}s")
print(f"  Params: {sum(p.numel() for p in model.parameters())}")
