import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

# 2. Positional Encoding pour Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=300, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x : (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# 3. Modèle Transformer pour classification de séries
class TransformerClassifier(nn.Module):
    def __init__(self,
                 in_channels=1,
                 d_model=64,
                 nhead=4,
                 num_layers=2,
                 num_classes=4,
                 max_len=300):
        super().__init__()
        # Project input → d_model
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        # Classifier après pooling
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x : (B, C=1, T) → (B, T, C)
        x = x.permute(0, 2, 1)
        # Project and add position
        x = self.input_proj(x)         # (B, T, d_model)
        x = self.pos_encoder(x)        # (B, T, d_model)
        x = self.transformer_encoder(x)  # (B, T, d_model)
        # Pool over time
        x = x.mean(dim=1)              # (B, d_model)
        return self.classifier(x)      # (B, num_classes)

# 4. Chargement des données
X0 = np.load("histo_PD_exp3_vs_epsilon.npy")  # (10000, 300)
X1 = np.load("histo_PD_exp3_vs_exp3.npy")     # (10000, 300)
X2 = np.load("histo_PD_exp3_vs_ucb.npy")      # (10000, 300)
X3 = np.load("histo_PD_exp3_vs_ftl.npy")      # (10000, 300)
X = np.vstack([X0, X1, X2, X3])               # (40000, 300)

# Si besoin, ne garder que les 200 derniers points :
# X = X[:, 100:]  # (40000, 200)

y = np.concatenate([
    np.zeros(len(X0), dtype=int),
    np.ones(len(X1),  dtype=int),
    2*np.ones(len(X2),  dtype=int),
    3*np.ones(len(X3),  dtype=int),
])                                            # (40000,)

# 5. Split train/val/test
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

# 6. Initialisation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model = TransformerClassifier(
    in_channels=1,
    d_model=64,
    nhead=4,
    num_layers=2,
    num_classes=4,
    max_len=X.shape[1]
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 7. Boucle d'entraînement / validation
num_epochs = 20
for epoch in range(1, num_epochs + 1):
    # Entraînement
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

    # Validation
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for Xb, yb in loader_val:
            Xb, yb = Xb.to(device), yb.to(device)
            val_correct += (model(Xb).argmax(1) == yb).sum().item()
    val_acc = val_correct / len(ds_val)

    print(f"Epoch {epoch:03d} — "
          f"train_loss: {train_loss:.4f}, "
          f"train_acc: {train_acc:.3f}, "
          f"val_acc:   {val_acc:.3f}")

# 8. Évaluation finale sur le test set
model.eval()
test_correct = 0
with torch.no_grad():
    for Xb, yb in loader_test:
        Xb, yb = Xb.to(device), yb.to(device)
        test_correct += (model(Xb).argmax(1) == yb).sum().item()
test_acc = test_correct / len(ds_test)
print(f"Test accuracy: {test_acc:.3f}")

# 9. Exemple de prédiction
sample_X, sample_y = ds_test[0]
sample_X = sample_X.unsqueeze(0).to(device)
model.eval()
with torch.no_grad():
    pred = model(sample_X).argmax(dim=1).item()
print(f"Vraie classe : {sample_y.item()}, Prédite : {pred}")

# 10. Matrice de confusion
y_true, y_pred = [], []
model.eval()
with torch.no_grad():
    for Xb, yb in loader_test:
        Xb = Xb.to(device)
        preds = model(Xb).argmax(dim=1).cpu().numpy()
        y_true.extend(yb.numpy())
        y_pred.extend(preds)

cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
print("Confusion matrix:\n", cm)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["exp3 vs ε", "exp3 vs exp3", "exp3 vs UCB", "exp3 vs FTL"]
)
fig, ax = plt.subplots(figsize=(6,6))
disp.plot(ax=ax, cmap="Blues")
plt.title("Matrice de confusion")
plt.show()

# 11. Pire confusion
cm_off = cm.copy()
np.fill_diagonal(cm_off, 0)
i, j = np.unravel_index(cm_off.argmax(), cm_off.shape)
print(f"Confusion maximale entre {i} → {j} : {cm_off[i,j]} fois")
