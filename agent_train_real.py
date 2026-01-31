#!/usr/bin/env python3
"""
Agent Zero — AI Training on REAL data
========================================
Dataset: sklearn digits — 1797 real handwritten digit images (8x8, 10 klase)
Ne generiše podatke sam. Koristi real data.

Evolution:
  Gen1: Linear classifier (no hidden layers)
  Gen2: MLP [64→32→10]
  Gen3: MLP [64→128→64→10] + Adam
  Gen4: MLP + Dropout + BatchNorm
  Gen5: CNN [Conv→Pool→Conv→Pool→FC] na 8x8
  Gen6: CNN + BatchNorm + LR scheduler + early stopping
  Gen7: ResNet-style skip connections
"""
import sys, os, json, subprocess, base64, re, time, urllib.request
from pathlib import Path

WORKSPACE = Path("/mnt/user-data/outputs/ai_training")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_OWNER = "pistakugli"
GITHUB_REPO = "agent-zero-ai-training"

# ============================================================
# SCAN
# ============================================================
def scan():
    gens = {}
    for f in sorted(WORKSPACE.glob("real_gen*_*.py")):
        m = re.match(r'real_gen(\d+)', f.stem)
        if m:
            num = int(m.group(1))
            code = f.read_text()
            gens[num] = {"file": f, "code": code}
            print(f"   📄 {f.name} ({len(code.splitlines())} lines)")
    return gens

# ============================================================
# ANALYZE
# ============================================================
def analyze(gens):
    result = {}
    for n, g in gens.items():
        c = g["code"]
        result[n] = {
            "hidden":     "nn.Linear" in c and c.count("nn.Linear") > 1,
            "adam":       "Adam" in c,
            "dropout":    "Dropout" in c,
            "batchnorm":  "BatchNorm" in c,
            "cnn":        "Conv2d" in c,
            "scheduler":  "ReduceLROnPlateau" in c or "StepLR" in c,
            "early_stop": "patience" in c,
            "resnet":     "skip" in c or "residual" in c,
        }
        r = result[n]
        if r["resnet"]: algo = "ResNet"
        elif r["cnn"] and r["batchnorm"]: algo = "CNN+BN+Sched"
        elif r["cnn"]: algo = "CNN"
        elif r["dropout"] and r["batchnorm"]: algo = "MLP+DO+BN"
        elif r["adam"] and r["hidden"]: algo = "MLP+Adam"
        elif r["hidden"]: algo = "MLP"
        else: algo = "Linear"
        result[n]["algo"] = algo
        print(f"   Gen{n}: {algo}")
    return result

# ============================================================
# DECIDE
# ============================================================
def decide_and_generate(analysis):
    if not analysis:
        print(f"   💭 Nema gen → Linear classifier")
        return write(1, "linear", CODE_GEN1)

    max_gen = max(analysis.keys())
    top = analysis[max_gen]
    next_gen = max_gen + 1

    if not top["hidden"]:
        print(f"   💭 Linear → MLP")
        return write(next_gen, "mlp", CODE_GEN2)
    elif top["hidden"] and not top["adam"]:
        print(f"   💭 MLP SGD → deeper MLP + Adam")
        return write(next_gen, "mlp_adam", CODE_GEN3)
    elif top["adam"] and not top["dropout"]:
        print(f"   💭 MLP+Adam → Dropout + BatchNorm")
        return write(next_gen, "mlp_dropout_bn", CODE_GEN4)
    elif top["dropout"] and not top["cnn"]:
        print(f"   💭 MLP → CNN")
        return write(next_gen, "cnn", CODE_GEN5)
    elif top["cnn"] and not top["batchnorm"]:
        print(f"   💭 CNN → CNN + BN + scheduler + early stop")
        return write(next_gen, "cnn_full", CODE_GEN6)
    elif top["cnn"] and top["batchnorm"] and not top["resnet"]:
        print(f"   💭 CNN → ResNet skip connections")
        return write(next_gen, "resnet", CODE_GEN7)
    else:
        print(f"   💭 Fully evolved")
        return None

def write(gen, label, template):
    code = template.format(gen=gen)
    name = f"real_gen{gen}_{label}.py"
    filepath = WORKSPACE / name
    filepath.write_text(code)
    print(f"   ✓ {name} ({len(code.splitlines())} lines)")
    return filepath

# ============================================================
# TEST
# ============================================================
def test(filepath):
    print(f"\n🧪 Training: {filepath.name}")
    result = subprocess.run(
        ["python3", str(filepath)],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode == 0:
        for line in result.stdout.strip().split('\n'):
            print(f"   {line}")
        return True
    else:
        print(f"   ✗ {result.stderr.strip()[-500:]}")
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
        "message": f"Agent Zero: {filepath.name}",
        "content": content,
        **({"sha": sha} if sha else {})
    }).encode()

    req = urllib.request.Request(url, data=data, headers=headers, method="PUT")
    try:
        urllib.request.urlopen(req)
        print(f"   ✓ pushed")
        return True
    except Exception as e:
        print(f"   ✗ {e}")
        return False

# ============================================================
# CODE — all use sklearn digits (REAL data)
# ============================================================

# Shared preamble used by all gens
DATA_PREAMBLE = '''
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
print(f"  Train: {{X_train_t.shape[0]}} | Val: {{X_val_t.shape[0]}} | Classes: 10")
'''

CODE_GEN1 = '''#!/usr/bin/env python3
"""Gen{gen} — Linear classifier on REAL handwritten digits.
No hidden layers. Just: input(64) → output(10).
Baseline to beat.
"""
''' + DATA_PREAMBLE + '''
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        return self.fc(x)

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

print(f"✓ Gen{gen} Linear [64→10]")
print(f"  Loss: {{loss.item():.4f}} | Val Acc: {{val_acc:.4f}} | Time: {{elapsed:.2f}}s")
print(f"  Params: {{sum(p.numel() for p in model.parameters())}}")
'''

CODE_GEN2 = '''#!/usr/bin/env python3
"""Gen{gen} — MLP [64→32→10]. Agent saw linear underfits → added hidden layer.
"""
''' + DATA_PREAMBLE + '''
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

print(f"✓ Gen{gen} MLP [64→32→10]")
print(f"  Loss: {{loss.item():.4f}} | Val Acc: {{val_acc:.4f}} | Time: {{elapsed:.2f}}s")
print(f"  Params: {{sum(p.numel() for p in model.parameters())}}")
'''

CODE_GEN3 = '''#!/usr/bin/env python3
"""Gen{gen} — Deeper MLP [64→128→64→10] + Adam.
Agent saw single hidden layer + SGD → went deeper + smarter optimizer.
"""
''' + DATA_PREAMBLE + '''
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.net(x)

model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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

print(f"✓ Gen{gen} MLP [64→128→64→10] + Adam")
print(f"  Loss: {{loss.item():.4f}} | Val Acc: {{val_acc:.4f}} | Time: {{elapsed:.2f}}s")
print(f"  Params: {{sum(p.numel() for p in model.parameters())}}")
'''

CODE_GEN4 = '''#!/usr/bin/env python3
"""Gen{gen} — MLP + Dropout + BatchNorm. Regularization.
Agent saw overfitting risk → added Dropout(0.3) + BatchNorm.
"""
''' + DATA_PREAMBLE + '''
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.net(x)

model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

start = time.time()
for epoch in range(200):
    model.train()
    out = model(X_train_t)
    loss = criterion(out, y_train_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
elapsed = time.time() - start

model.eval()
with torch.no_grad():
    val_acc = (model(X_val_t).argmax(1) == y_val_t).float().mean().item()

print(f"✓ Gen{gen} MLP + Dropout + BatchNorm")
print(f"  Loss: {{loss.item():.4f}} | Val Acc: {{val_acc:.4f}} | Time: {{elapsed:.2f}}s")
print(f"  Params: {{sum(p.numel() for p in model.parameters())}}")
'''

CODE_GEN5 = '''#!/usr/bin/env python3
"""Gen{gen} — CNN on 8x8 digit images.
Agent saw MLP treats pixels as flat vector → reshape to 1x8x8, use Conv2d.
Spatial structure matters for digits.
"""
''' + DATA_PREAMBLE + '''
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

print(f"✓ Gen{gen} CNN [Conv32→Pool→Conv64→Pool→FC]")
print(f"  Loss: {{loss.item():.4f}} | Val Acc: {{val_acc:.4f}} | Time: {{elapsed:.2f}}s")
print(f"  Params: {{sum(p.numel() for p in model.parameters())}}")
'''

CODE_GEN6 = '''#!/usr/bin/env python3
"""Gen{gen} — CNN + BatchNorm + LR scheduler + early stopping.
Production-grade training loop. Agent saw basic CNN → added full pipeline.
"""
''' + DATA_PREAMBLE + '''
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

print(f"✓ Gen{gen} CNN + BN + Scheduler + EarlyStopping")
print(f"  Loss: {{loss.item():.4f}} | Best Val Acc: {{best_val_acc:.4f}} | Epochs: {{epoch+1}}")
print(f"  Time: {{elapsed:.2f}}s | Final LR: {{optimizer.param_groups[0]['lr']:.6f}}")
print(f"  Params: {{sum(p.numel() for p in model.parameters())}}")
'''

CODE_GEN7 = '''#!/usr/bin/env python3
"""Gen{gen} — ResNet-style with skip connections.
Agent saw CNN saturate → added residual/skip connections.
Skip connection: output = F(x) + x — gradient flows directly.
"""
''' + DATA_PREAMBLE + '''
X_train_img = X_train_t.view(-1, 1, 8, 8)
X_val_img   = X_val_t.view(-1, 1, 8, 8)

class ResBlock(nn.Module):
    """Residual block: F(x) + x"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
    def forward(self, x):
        # skip connection
        return torch.relu(self.block(x) + x)

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.res1 = ResBlock(32)                          # skip
        self.pool1 = nn.MaxPool2d(2)                      # 8→4
        self.expand = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.res2 = ResBlock(64)                          # skip
        self.pool2 = nn.MaxPool2d(2)                      # 4→2
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.pool1(x)
        x = self.expand(x)
        x = self.res2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = ResNet()
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

print(f"✓ Gen{gen} ResNet [stem→ResBlock→pool→expand→ResBlock→pool→FC]")
print(f"  Loss: {{loss.item():.4f}} | Best Val Acc: {{best_val_acc:.4f}} | Epochs: {{epoch+1}}")
print(f"  Time: {{elapsed:.2f}}s | Params: {{sum(p.numel() for p in model.parameters())}}")
print(f"  Skip connections: 2")
'''

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("AGENT ZERO — AI TRAINING ON REAL DATA")
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
            code = filepath.read_text()
            num = max(analysis.keys()) + 1 if analysis else 1
            analysis[num] = {
                "hidden":     "nn.Linear" in code and code.count("nn.Linear") > 1,
                "adam":       "Adam" in code,
                "dropout":    "Dropout" in code,
                "batchnorm":  "BatchNorm" in code,
                "cnn":        "Conv2d" in code,
                "scheduler":  "ReduceLROnPlateau" in code,
                "early_stop": "patience" in code,
                "resnet":     "skip" in code or "ResBlock" in code,
            }
        else:
            print("   ⚠ Failed - stop.")
            break

    print(f"\n{'=' * 60}")
    print("✅ DONE")
    print(f"{'=' * 60}")
