# train.py
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from .model import TimeSeriesDataset, TransformerClassifier, SimpleCNN1D
import zlib
import lzma
import bz2
from collections import Counter
import math

def approx__complexity(seq): 
    # Convertir la séquence en bytes
    data = bytes(seq)
    
    # Compresser avec différents algorithmes
    comp_zlib = len(zlib.compress(data))
    comp_lzma = len(lzma.compress(data))
    comp_bz2  = len(bz2.compress(data))
    
    # Choisir la plus petite taille compressée
    min_bytes = min(comp_zlib, comp_lzma, comp_bz2)
    
    # Retourner la complexité normalisée (bits par symbole)
    return (min_bytes * 8) / len(seq)


def calcul_exactitude_base(Z0, Z1, Z2, Z3):

    somme_Z0 = np.sum(Z0,axis=1)
    somme_Z1 = np.sum(Z1,axis=1)
    somme_Z2 = np.sum(Z2,axis=1)
    somme_Z3 = np.sum(Z3,axis=1)

    all_data = np.concatenate([somme_Z0, somme_Z1, somme_Z2, somme_Z3])
    x_grid = np.linspace(all_data.min() - 1, all_data.max() + 1, 300)
    """
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
    """
    try:
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
    except:
        accuracy=0
    return accuracy


def train(X0, X1, X2, X3, modele):
    # 1. Empiler les données et créer les étiquettes
    X = np.vstack([X0, X1, X2, X3])
    y = np.concatenate([
        np.zeros(len(X0), dtype=int),
        np.ones(len(X1),  dtype=int),
        2 * np.ones(len(X2),  dtype=int),
        3 * np.ones(len(X3),  dtype=int),
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
    if modele == "cnn":
        model = SimpleCNN1D(in_channels=1, num_classes=4).to(device)
    if modele == "transformer":
        model = TransformerClassifier(in_channels=1, num_classes=4).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 4. Entraînement / validation
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

    return test_acc

# 1. Chargement des données
def load_data_300(algo):
    X0 = np.load(f"histo_PD_{algo}_vs_epsilon_v5.npy")
    X1 = np.load(f"histo_PD_{algo}_vs_exp3_v5.npy")
    X2 = np.load(f"histo_PD_{algo}_vs_ucb_v5.npy")
    X3 = np.load(f"histo_PD_{algo}_vs_ftl_v5.npy")
    Z0 = np.load(f"histo_r_PD_{algo}_vs_epsilon_v5.npy")
    Z1 = np.load(f"histo_r_PD_{algo}_vs_exp3_v5.npy")
    Z2 = np.load(f"histo_r_PD_{algo}_vs_ucb_v5.npy")
    Z3 = np.load(f"histo_r_PD_{algo}_vs_ftl_v5.npy")
    return X0, X1, X2, X3, Z0, Z1, Z2, Z3

def load_data_150(algo):
    X0 = np.load(f"histo_PD_{algo}_vs_epsilon_v5.npy")[:,150:]
    X1 = np.load(f"histo_PD_{algo}_vs_exp3_v5.npy")[:,150:]
    X2 = np.load(f"histo_PD_{algo}_vs_ucb_v5.npy")[:,150:]
    X3 = np.load(f"histo_PD_{algo}_vs_ftl_v5.npy")[:,150:]
    Z0 = np.load(f"histo_r_PD_{algo}_vs_epsilon_v5.npy")[:,150:]
    Z1 = np.load(f"histo_r_PD_{algo}_vs_exp3_v5.npy")[:,150:]
    Z2 = np.load(f"histo_r_PD_{algo}_vs_ucb_v5.npy")[:,150:]
    Z3 = np.load(f"histo_r_PD_{algo}_vs_ftl_v5.npy")[:,150:]
    return X0, X1, X2, X3, Z0, Z1, Z2, Z3







def train2(longueur, algo):
    if longueur==150:
        X0, X1, X2, X3, Z0, Z1, Z2, Z3 = load_data_150(algo)
    else: 
        X0, X1, X2, X3, Z0, Z1, Z2, Z3 = load_data_300(algo)
    print("resu")
    exact_test_cnn = train(X0, X1, X2, X3, "cnn")
    exact_test_transformer = train(X0, X1, X2, X3, "transformer")
    exact_base = calcul_exactitude_base(Z0, Z1, Z2, Z3)

    result0 = np.apply_along_axis(approx__complexity, axis=1, arr=X0)
    c0 = np.mean(result0)
    s0 = np.std(result0)

    result1 = np.apply_along_axis(approx__complexity, axis=1, arr=X1)
    c1 = np.mean(result1)
    s1 = np.std(result1)

    result2 = np.apply_along_axis(approx__complexity, axis=1, arr=X2)
    c2 = np.mean(result2)
    s2 = np.std(result2)

    result3 = np.apply_along_axis(approx__complexity, axis=1, arr=X3)
    c3 = np.mean(result3)
    s3 = np.std(result3)


    return longueur, algo, exact_test_cnn, exact_test_transformer, float(exact_base), c0, c1, c2, c3, s0, s1, s2, s3


# 1) On collecte tous les résultats
results = []
for longueur in (300, 150):
    for algo in ("exp3", "ucb", "ftl", "epsilon"):
        longueur_, algo_, acc_cnn, acc_trans, acc_base, c0, c1, c2, c3, s0, s1, s2, s3 = train2(longueur, algo)
        results.append({
            "longueur": longueur_,
            "algo":     algo_,
            "acc_cnn":  acc_cnn,
            "acc_trans": acc_trans,
            "acc_base": acc_base,
            "epsilon_m": c0,
            "exp3_m": c1,
            "ucb_m": c2,
            "ftl_m": c3,
            "epsilon_s": s0,
            "exp3_s": s1,
            "ucb_s": s2,
            "ftl_s": s3
        })

# 2) On construit un DataFrame et on sauve
df = pd.DataFrame(results)
df.to_csv("experiment_results.csv", index=False)
print("Saved to experiment_results.csv")
