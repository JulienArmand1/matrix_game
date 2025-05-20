# train.py
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

from .model import TimeSeriesDataset, TransformerClassifier, SimpleCNN1D
# 1. Chargement des données
X0 = np.load("histo_PD_exp3_vs_epsilon_v3.npy")[:,150:]
X1 = np.load("histo_PD_exp3_vs_exp3_v3.npy")[:,150:]
X2 = np.load("histo_PD_exp3_vs_ucb_v3.npy")[:,150:]
X3 = np.load("histo_PD_exp3_vs_ftl_v3.npy")[:,150:]

Z0 = np.load("histo_r_PD_exp3_vs_epsilon_v3.npy")[:,150:]
Z1 = np.load("histo_r_PD_exp3_vs_exp3_v3.npy")[:,150:]
Z2 = np.load("histo_r_PD_exp3_vs_ucb_v3.npy")[:,150:]
Z3 = np.load("histo_r_PD_exp3_vs_ftl_v3.npy")[:,150:]


somme_Z0 = np.sum(Z0,axis=1)
somme_Z1 = np.sum(Z1,axis=1)
somme_Z2 = np.sum(Z2,axis=1)
somme_Z3 = np.sum(Z3,axis=1)
"""
plt.figure()
plt.hist(somme_Z0, bins=30, alpha=0.5, label='Distribution 1', color='blue')
plt.hist(somme_Z1, bins=30, alpha=0.5, label='Distribution 2', color='orange')
plt.hist(somme_Z2, bins=30, alpha=0.5, label='Distribution 3', color='green')
plt.hist(somme_Z3, bins=30, alpha=0.5, label='Distribution 4', color='red')
plt.title('Distribution des données')
plt.xlabel('Valeur')
plt.ylabel('Fréquence')
"""


all_data = np.concatenate([somme_Z0, somme_Z1, somme_Z2, somme_Z3])
x_grid = np.linspace(all_data.min() - 1, all_data.max() + 1, 300)

plt.figure()
for data, label in zip([somme_Z0, somme_Z1, somme_Z2, somme_Z3],
                       ['histo_r_PD_exp3_vs_epsilon_v3', 'histo_r_PD_exp3_vs_exp3_v3', 'histo_r_PD_exp3_vs_ucb_v3.npy', 'histo_r_PD_exp3_vs_ftl_v3.npy']):
    kde = gaussian_kde(data)
    plt.plot(x_grid, kde(x_grid), label=label)
    
plt.title('gaussian_kde des 4 distributions')
plt.xlabel('Valeur')
plt.ylabel('Densité')
plt.legend()
plt.show()

kdes = [
    gaussian_kde(somme_Z0),
    gaussian_kde(somme_Z1),
    gaussian_kde(somme_Z2),
    gaussian_kde(somme_Z3)
]

# Création des données combinées et des étiquettes vraies
X = np.concatenate([somme_Z0, somme_Z1, somme_Z2, somme_Z3])
y_true = np.concatenate([
    np.zeros_like(somme_Z0, dtype=int),
    np.ones_like(somme_Z1, dtype=int),
    np.full_like(somme_Z2, 2, dtype=int),
    np.full_like(somme_Z3, 3, dtype=int)
])

# Prédiction : distribution la plus probable pour chaque point
densities = np.vstack([kde(X) for kde in kdes])
y_pred = np.argmax(densities, axis=0)

# Calcul de l'exactitude
accuracy = np.mean(y_pred == y_true)
print(f"Accuracy (exactitude) : {accuracy * 100:.2f}%")



X  = np.vstack([X0, X1, X2, X3])
y  = np.concatenate([
    np.zeros(len(X0), dtype=int),
    np.ones(len(X1),  dtype=int),
    2*np.ones(len(X2),  dtype=int),
    3*np.ones(len(X3),  dtype=int),
])

# 2. Split train/val/test
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

# 3. Initialisation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model = SimpleCNN1D(
    in_channels=1,
    num_classes=4
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. Entraînement / validation
num_epochs = 10
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

    print(f"Epoch {epoch:02d} — train_loss: {train_loss:.4f}, "
          f"train_acc: {train_acc:.3f}, val_acc: {val_acc:.3f}")

# 5. Évaluation finale
model.eval()
test_correct = 0
with torch.no_grad():
    for Xb, yb in loader_test:
        Xb, yb = Xb.to(device), yb.to(device)
        test_correct += (model(Xb).argmax(1) == yb).sum().item()
test_acc = test_correct / len(ds_test)
print(f"Test accuracy: {test_acc:.3f}")

print(model)

# 6. Matrice de confusion
y_true, y_pred = [], []
with torch.no_grad():
    for Xb, yb in loader_test:
        Xb = Xb.to(device)
        preds = model(Xb).argmax(1).cpu().numpy()
        y_true.extend(yb.numpy())
        y_pred.extend(preds)

cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["exp3 vs ε", "exp3 vs exp3", "exp3 vs UCB", "exp3 vs FTL"]
)
fig, ax = plt.subplots(figsize=(6,6))
disp.plot(ax=ax, cmap="Blues")
plt.title("Matrice de confusion")
plt.show()
