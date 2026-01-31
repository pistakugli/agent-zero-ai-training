#!/usr/bin/env python3
"""Gen5 — Deep MLP [2→64→32→1] + Dropout(0.3).
Agent saw single hidden layer + Adam → went deeper + added regularization.
Harder dataset: XOR-like nonlinear boundary.
"""
import torch
import torch.nn as nn
import numpy as np
import time

np.random.seed(42)
N = 2000
# XOR-like: nonlinear boundary
X = np.random.randn(N, 2)
y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.float64)

X_train = torch.tensor(X[:1600], dtype=torch.float32)
y_train = torch.tensor(y[:1600], dtype=torch.float32).unsqueeze(1)
X_val   = torch.tensor(X[1600:], dtype=torch.float32)
y_val   = torch.tensor(y[1600:], dtype=torch.float32)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

start = time.time()
for epoch in range(200):
    model.train()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

elapsed = time.time() - start

model.eval()
with torch.no_grad():
    val_pred = model(X_val)
    val_acc = ((val_pred > 0.5).squeeze() == y_val).float().mean().item()

print(f"✓ Gen5 Deep MLP [2→64→32→1] + Dropout")
print(f"  Loss: {loss.item():.4f} | Val Accuracy: {val_acc:.4f} | Time: {elapsed:.3f}s")
print(f"  Params: {sum(p.numel() for p in model.parameters())}")
print(f"  Dataset: XOR (nonlinear)")
