
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# DATA — Kaggle digit-recognizer competition format
# ============================================================
train_df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test_df  = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

# Parse
y_train = train_df.iloc[:, 0].values
X_train = train_df.iloc[:, 1:].values.astype(np.float32) / 255.0
X_test  = test_df.values.astype(np.float32) / 255.0

# Split train → train/val
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Reshape → (N, 1, 28, 28)
X_tr  = torch.tensor(X_tr).view(-1, 1, 28, 28).to(device)
y_tr  = torch.tensor(y_tr, dtype=torch.long).to(device)
X_val = torch.tensor(X_val).view(-1, 1, 28, 28).to(device)
y_val = torch.tensor(y_val, dtype=torch.long).to(device)
X_test_t = torch.tensor(X_test).view(-1, 1, 28, 28).to(device)

print(f"Train: {X_tr.shape} | Val: {X_val.shape} | Test: {X_test_t.shape}")

train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=128, shuffle=True)

# ============================================================
# MODEL — ResNet-style CNN
# ============================================================
class ResBlock(nn.Module):
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
        return torch.relu(self.block(x) + x)  # skip connection

class MNISTResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            nn.MaxPool2d(2),          # 28→14
        )
        self.expand1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
            nn.MaxPool2d(2),          # 14→7
        )
        self.expand2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            ResBlock(256),
            nn.AdaptiveAvgPool2d(1),  # 7→1
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.expand1(x)
        x = self.layer2(x)
        x = self.expand2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = MNISTResNet().to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# TRAINING
# ============================================================
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0
patience = 7
no_improve = 0

start = time.time()
for epoch in range(50):
    # Train
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        out = model(X_batch)
        loss = criterion(out, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Val
    model.eval()
    with torch.no_grad():
        val_out = model(X_val)
        val_loss = criterion(val_out, y_val).item()
        val_acc = (val_out.argmax(1) == y_val).float().mean().item()
    
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "/kaggle/working/best_model.pth")
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

elapsed = time.time() - start
print(f"\nBest Val Acc: {best_val_acc:.4f} | Time: {elapsed:.1f}s")

# ============================================================
# PREDICT + SUBMIT
# ============================================================
model.load_state_dict(torch.load("/kaggle/working/best_model.pth"))
model.eval()

with torch.no_grad():
    # Predict in batches
    preds = []
    for i in range(0, len(X_test_t), 256):
        batch = X_test_t[i:i+256]
        preds.append(model(batch).argmax(1).cpu().numpy())
    preds = np.concatenate(preds)

submission = pd.DataFrame({"ImageId": range(1, len(preds)+1), "Label": preds})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print(f"\n✓ submission.csv saved — {len(preds)} predictions")
print(f"  Distribution: {dict(zip(*np.unique(preds, return_counts=True)))}")

