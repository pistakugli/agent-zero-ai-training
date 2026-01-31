#!/usr/bin/env python3
"""Gen2 — PyTorch linear classifier. Autograd replaces manual backprop.
Same architecture as Gen1, but torch handles gradients.
"""
import torch
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

# Model — linear
W = torch.zeros(2, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.1

start = time.time()
for epoch in range(100):
    z = X_train @ W + b
    pred = torch.sigmoid(z)
    loss = torch.nn.functional.binary_cross_entropy(pred, y_train)
    
    loss.backward()
    with torch.no_grad():
        W -= lr * W.grad
        b -= lr * b.grad
        W.grad.zero_()
        b.grad.zero_()

elapsed = time.time() - start

with torch.no_grad():
    val_pred = torch.sigmoid(X_val @ W + b)
    val_acc = ((val_pred > 0.5).squeeze() == y_val).float().mean().item()

print(f"✓ Gen2 PyTorch Linear (autograd)")
print(f"  Loss: {loss.item():.4f} | Val Accuracy: {val_acc:.4f} | Time: {elapsed:.3f}s")
