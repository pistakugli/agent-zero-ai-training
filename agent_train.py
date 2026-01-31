#!/usr/bin/env python3
"""
Agent Zero — Autonomous AI Training Evolution
================================================
Agent sam generiše modele, trenira ih, meri perormanse,
i evolutionira arhitekturu.

Evolution path:
  Gen1: Pure NumPy — manual forward + backprop
  Gen2: PyTorch autograd — ista arhitektura, torch tensori
  Gen3: Hidden layer + ReLU
  Gen4: Adam optimizer
  Gen5: Multi-layer + Dropout
  Gen6: CNN na sintetičkim image data
  Gen7: BatchNorm + LR scheduler + early stopping

Dataset: synthetic — isti seed svaka put → fair comparison.
"""

import sys, os, json, subprocess, base64, re, time, urllib.request
from pathlib import Path

WORKSPACE = Path("/mnt/user-data/outputs/ai_training")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "YOUR_TOKEN")
GITHUB_OWNER = "pistakugli"
GITHUB_REPO = "agent-zero-ai-training"

# ============================================================
# SCAN — čita sve postoji generacije
# ============================================================
def scan():
    gens = {}
    for f in sorted(WORKSPACE.glob("gen*_*.py")):
        m = re.match(r'gen(\d+)', f.stem)
        if m:
            num = int(m.group(1))
            code = f.read_text()
            gens[num] = {"file": f, "code": code}
            print(f"   📄 {f.name} ({len(code.splitlines())} lines)")
    return gens

# ============================================================
# ANALYZE — šta svaka generacija ima
# ============================================================
def analyze(gens):
    result = {}
    for n, g in gens.items():
        c = g["code"]
        result[n] = {
            "autograd":   "torch" in c and "backward()" in c,
            "hidden":     "hidden" in c.lower() or "nn.Linear" in c,
            "adam":       "Adam" in c,
            "dropout":    "Dropout" in c,
            "cnn":        "Conv2d" in c,
            "batchnorm":  "BatchNorm" in c,
            "scheduler":  "scheduler" in c.lower() or "StepLR" in c or "ReduceLR" in c,
            "early_stop": "early_stop" in c or "patience" in c,
        }
        r = result[n]
        if r["cnn"] and r["batchnorm"]: algo = "CNN+BN+Scheduler"
        elif r["cnn"]: algo = "CNN"
        elif r["dropout"]: algo = "MLP+Dropout"
        elif r["adam"] and r["hidden"]: algo = "MLP+Adam"
        elif r["hidden"]: algo = "MLP+SGD"
        elif r["autograd"]: algo = "Linear+Torch"
        else: algo = "Linear+NumPy"
        result[n]["algo"] = algo
        print(f"   Gen{n}: {algo}")
    return result

# ============================================================
# DECIDE — šta generiše sledeće
# ============================================================
def decide_and_generate(analysis):
    if not analysis:
        # Prva generacija
        print(f"   💭 Nema generacija → Gen1 pure NumPy")
        return write(1, "numpy_linear", CODE_GEN1_NUMPY)

    max_gen = max(analysis.keys())
    top = analysis[max_gen]
    next_gen = max_gen + 1

    if not top["autograd"]:
        print(f"   💭 NumPy manual backprop → PyTorch autograd")
        return write(next_gen, "torch_linear", CODE_GEN2_TORCH)
    elif top["autograd"] and not top["hidden"]:
        print(f"   💭 Linear → Hidden layer + ReLU")
        return write(next_gen, "mlp_relu", CODE_GEN3_MLP)
    elif top["hidden"] and not top["adam"]:
        print(f"   💭 SGD → Adam optimizer")
        return write(next_gen, "mlp_adam", CODE_GEN4_ADAM)
    elif top["adam"] and not top["dropout"]:
        print(f"   💭 Adam → Multi-layer + Dropout")
        return write(next_gen, "mlp_dropout", CODE_GEN5_DROPOUT)
    elif top["dropout"] and not top["cnn"]:
        print(f"   💭 MLP → CNN na image data")
        return write(next_gen, "cnn", CODE_GEN6_CNN)
    elif top["cnn"] and not top["batchnorm"]:
        print(f"   💭 CNN → BatchNorm + LR scheduler + early stopping")
        return write(next_gen, "cnn_full", CODE_GEN7_CNN_FULL)
    else:
        print(f"   💭 Fully evolved")
        return None

def write(gen, label, template):
    code = template.format(gen=gen)
    name = f"gen{gen}_{label}.py"
    filepath = WORKSPACE / name
    filepath.write_text(code)
    print(f"   ✓ {name} ({len(code.splitlines())} lines)")
    return filepath

# ============================================================
# TEST — trenira model, vraća rezultate
# ============================================================
def test(filepath):
    print(f"\n🧪 Training: {filepath.name}")
    result = subprocess.run(
        ["python3", str(filepath)],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode == 0:
        for line in result.stdout.strip().split('\n'):
            print(f"   {line}")
        return True
    else:
        print(f"   ✗ ERROR:")
        print(f"   {result.stderr.strip()[-300:]}")
        return False

# ============================================================
# PUSH
# ============================================================
def push(filepath):
    print(f"\n⬆️  Push: {filepath.name}")
    content = base64.b64encode(filepath.read_bytes()).decode()
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{filepath.name}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json"
    }
    sha = None
    try:
        req = urllib.request.Request(url, headers={k:v for k,v in headers.items() if k != "Content-Type"})
        sha = json.loads(urllib.request.urlopen(req).read())["sha"]
    except: pass

    data = json.dumps({
        "message": f"Agent Zero evolved: {filepath.name}",
        "content": content,
        **({"sha": sha} if sha else {})
    }).encode()

    req = urllib.request.Request(url, data=data, headers=headers, method="PUT")
    try:
        urllib.request.urlopen(req)
        print(f"   ✓ https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}")
        return True
    except Exception as e:
        print(f"   ✗ {e}")
        return False

# ============================================================
# CODE TEMPLATES
# ============================================================

# --- Gen1: Pure NumPy, manual forward + backprop ---
CODE_GEN1_NUMPY = '''#!/usr/bin/env python3
"""Gen{gen} — Pure NumPy linear classifier. Manual forward + backprop.
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

print(f"✓ Gen{gen} NumPy Linear")
print(f"  Loss: {{loss:.4f}} | Val Accuracy: {{val_acc:.4f}} | Time: {{elapsed:.3f}}s")
print(f"  Params: W={{W.flatten()}} b={{b}}")
'''

# --- Gen2: Same task, PyTorch autograd ---
CODE_GEN2_TORCH = '''#!/usr/bin/env python3
"""Gen{gen} — PyTorch linear classifier. Autograd replaces manual backprop.
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

print(f"✓ Gen{gen} PyTorch Linear (autograd)")
print(f"  Loss: {{loss.item():.4f}} | Val Accuracy: {{val_acc:.4f}} | Time: {{elapsed:.3f}}s")
'''

# --- Gen3: Hidden layer + ReLU ---
CODE_GEN3_MLP = '''#!/usr/bin/env python3
"""Gen{gen} — MLP: input → hidden(16) → ReLU → output.
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

print(f"✓ Gen{gen} MLP [2→16→1] + ReLU")
print(f"  Loss: {{loss.item():.4f}} | Val Accuracy: {{val_acc:.4f}} | Time: {{elapsed:.3f}}s")
print(f"  Params: {{sum(p.numel() for p in model.parameters())}}")
'''

# --- Gen4: Adam ---
CODE_GEN4_ADAM = '''#!/usr/bin/env python3
"""Gen{gen} — MLP + Adam optimizer. Agent saw SGD → replaced with Adam.
Adam: adaptive learning rates per parameter. Converges faster.
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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

print(f"✓ Gen{gen} MLP + Adam")
print(f"  Loss: {{loss.item():.4f}} | Val Accuracy: {{val_acc:.4f}} | Time: {{elapsed:.3f}}s")
'''

# --- Gen5: Deeper + Dropout ---
CODE_GEN5_DROPOUT = '''#!/usr/bin/env python3
"""Gen{gen} — Deep MLP [2→64→32→1] + Dropout(0.3).
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

print(f"✓ Gen{gen} Deep MLP [2→64→32→1] + Dropout")
print(f"  Loss: {{loss.item():.4f}} | Val Accuracy: {{val_acc:.4f}} | Time: {{elapsed:.3f}}s")
print(f"  Params: {{sum(p.numel() for p in model.parameters())}}")
print(f"  Dataset: XOR (nonlinear)")
'''

# --- Gen6: CNN ---
CODE_GEN6_CNN = '''#!/usr/bin/env python3
"""Gen{gen} — CNN on synthetic image data.
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

print(f"✓ Gen{gen} CNN [Conv2d→Pool→Conv2d→Pool→FC]")
print(f"  Loss: {{loss.item():.4f}} | Val Accuracy: {{val_acc:.4f}} | Time: {{elapsed:.3f}}s")
print(f"  Params: {{sum(p.numel() for p in model.parameters())}}")
print(f"  Dataset: synthetic 28x28 stripes")
'''

# --- Gen7: CNN + BatchNorm + LR Scheduler + Early Stopping ---
CODE_GEN7_CNN_FULL = '''#!/usr/bin/env python3
"""Gen{gen} — Full CNN pipeline: BatchNorm + LR scheduling + early stopping.
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

print(f"✓ Gen{gen} CNN+BatchNorm+Scheduler+EarlyStopping")
print(f"  Loss: {{loss.item():.4f}} | Best Val Acc: {{best_val_acc:.4f}} | Epochs: {{epoch+1}}")
print(f"  Time: {{elapsed:.3f}}s | Final LR: {{final_lr:.6f}}")
print(f"  Params: {{sum(p.numel() for p in model.parameters())}}")
print(f"  Dataset: 4-class synthetic 28x28")
'''

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("AGENT ZERO — AI TRAINING EVOLUTION")
    print("=" * 60)

    print("\n📖 Scan:")
    gens = scan()

    print("\n🔍 Analyze:")
    analysis = analyze(gens)

    print("\n🧬 Evolution:")
    for i in range(7):
        print(f"\n{'─'*60}")
        filepath = decide_and_generate(analysis)
        if filepath is None:
            print("   Fully evolved.")
            break

        if test(filepath):
            push(filepath)
            # Update analysis for next iteration
            code = filepath.read_text()
            num = max(analysis.keys()) + 1 if analysis else 1
            analysis[num] = {
                "autograd":   "torch" in code and "backward()" in code,
                "hidden":     "hidden" in code.lower() or "nn.Linear" in code,
                "adam":       "Adam" in code,
                "dropout":    "Dropout" in code,
                "cnn":        "Conv2d" in code,
                "batchnorm":  "BatchNorm" in code,
                "scheduler":  "scheduler" in code.lower() or "ReduceLR" in code,
                "early_stop": "early_stop" in code or "patience" in code,
                "algo":       "evolved"
            }
        else:
            print("   ⚠ Failed - stop.")
            break

    print(f"\n{'=' * 60}")
    print("✅ DONE")
    print(f"{'=' * 60}")
