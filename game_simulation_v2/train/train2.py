import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# 1. Dataset pour séries temporelles
class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert len(X) == len(y)
        if X.ndim == 2:
            X = X[:, None, :]  # (N, 1, T)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 2. Modèle 1D-CNN simple
class SimpleCNN1D(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(1),  # (B,32,1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),               # (B,32)
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_classes)  # (B,2)
        )

    def forward(self, x):
        h = self.feature_extractor(x)  # (B,32,1)
        return self.classifier(h)      # (B,2)

# 3. Chargement des données depuis deux fichiers .npy, un par classe
X0 = np.load("histo_PD_exp3_vs_epsilon.npy")  # (10000, 300)
X1 = np.load("histo_PD_exp3_vs_exp3.npy")     # (10000, 300)
X2 = np.load("histo_PD_exp3_vs_ucb.npy")  # (10000, 300)
X3 = np.load("histo_PD_exp3_vs_ftl.npy")     # (10000, 300)
X = np.vstack([X0, X1, X2, X3])                      # (40000, 300)
y = np.concatenate([
    np.zeros(len(X0), dtype=int),
    np.ones(len(X1),  dtype=int),
    2*np.ones(len(X2),  dtype=int),
    3*np.ones(len(X3),  dtype=int)
])                                            # (40000,)

# 4. Split train/validation/test (70% / 15% / 15%)
N = len(y)
perm = np.random.permutation(N)
train_end = int(0.70 * N)
val_end   = int(0.85 * N)

train_idx = perm[:train_end]
val_idx   = perm[train_end:val_end]
test_idx  = perm[val_end:]

ds_train = TimeSeriesDataset(X[train_idx], y[train_idx])
ds_val   = TimeSeriesDataset(X[val_idx],   y[val_idx])
ds_test  = TimeSeriesDataset(X[test_idx],  y[test_idx])

loader_train = DataLoader(ds_train, batch_size=32, shuffle=True)
loader_val   = DataLoader(ds_val,   batch_size=64, shuffle=False)
loader_test  = DataLoader(ds_test,  batch_size=64, shuffle=False)

# 5. Initialisation du modèle, de l'optimiseur et de la loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = SimpleCNN1D(in_channels=1, num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 6. Boucle d'entraînement / validation
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    # — Entraînement —
    model.train()
    total_loss, total_correct = 0.0, 0
    for Xb, yb in loader_train:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
        total_correct += (logits.argmax(1) == yb).sum().item()
    train_loss = total_loss / len(ds_train)
    train_acc  = total_correct / len(ds_train)

    # — Validation —
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for Xb, yb in loader_val:
            Xb, yb = Xb.to(device), yb.to(device)
            val_correct += (model(Xb).argmax(1) == yb).sum().item()
    val_acc = val_correct / len(ds_val)

    print(f"Époch {epoch:02d} — "
          f"train_loss: {train_loss:.4f}, "
          f"train_acc: {train_acc:.3f}, "
          f"val_acc:   {val_acc:.3f}")

# 7. Évaluation finale sur le test set
model.eval()
test_correct = 0
with torch.no_grad():
    for Xb, yb in loader_test:
        Xb, yb = Xb.to(device), yb.to(device)
        test_correct += (model(Xb).argmax(1) == yb).sum().item()
test_acc = test_correct / len(ds_test)
print(f"Test accuracy: {test_acc:.3f}")


sample_X, sample_y = ds_test[0]           # sample_X : (1,T) tensor, sample_y : label
sample_X = sample_X.unsqueeze(0).to(device)  # ajouter la dimension batch → (1,1,T)

model.eval()
with torch.no_grad():
    logits = model(sample_X)              # (1,4)
    pred = logits.argmax(dim=1).item()    # indice de la classe prédite

# Affichage
print(f"Étiquette réelle : {sample_y.item()}") 
print(f"Prédiction du modèle : {pred}")