#!/usr/bin/env python3
"""Gen1 — Pure NumPy linear classifier. Manual forward + backprop.
No framework. Understand everything from scratch.
Dataset: synthetic 2-class, 2D features.
"""
import numpy as np
import time

# Synthetic data — fixed seed
np.random.seed(42)
N = 1000
X = np.random.randn(N, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)  # linear boundary

# Train/val split
X_train, X_val = X[:800], X[800:]
y_train, y_val = y[:800], y[800:]

# Weights
W = np.zeros((2, 1))
b = np.zeros((1,))
lr = 0.1

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

start = time.time()
for epoch in range(100):
    # Forward
    z = X_train @ W + b
    pred = sigmoid(z)
    
    # Loss (binary cross entropy)
    loss = -np.mean(y_train.reshape(-1,1) * np.log(pred + 1e-8) + (1 - y_train.reshape(-1,1)) * np.log(1 - pred + 1e-8))
    
    # Backprop — manual gradients
    error = pred - y_train.reshape(-1, 1)
    dW = X_train.T @ error / N
    db = np.mean(error)
    
    # Update
    W -= lr * dW
    b -= lr * db

elapsed = time.time() - start

# Evaluate
val_pred = sigmoid(X_val @ W + b)
val_acc = np.mean((val_pred > 0.5).flatten() == y_val)

print(f"✓ Gen1 NumPy Linear")
print(f"  Loss: {loss:.4f} | Val Accuracy: {val_acc:.4f} | Time: {elapsed:.3f}s")
print(f"  Params: W={W.flatten()} b={b}")
