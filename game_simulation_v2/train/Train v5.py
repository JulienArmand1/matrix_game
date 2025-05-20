import math
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = None):
        if X.ndim == 2:
            X = X[:, None, :]
        if seq_len:
            X = X[:, :, -seq_len:]
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(patterns, seq_len=None):
    data, labels = [], []
    for idx, pattern in enumerate(patterns):
        for path in sorted(Path().glob(pattern)):
            arr = np.load(path)
            data.append(arr)
            labels.append(np.full(len(arr), idx, dtype=int))
    X = np.vstack(data)
    y = np.concatenate(labels)
    return X, y


def get_loaders(X, y, batch=64, val_frac=0.15, test_frac=0.15, seq_len=None):
    ds = TimeSeriesDataset(X, y, seq_len)
    n = len(ds)
    splits = [n - int(frac * n) for frac in (val_frac + test_frac, test_frac)]
    train_ds, val_ds, test_ds = random_split(ds, splits)
    return (
        DataLoader(train_ds, batch_size=batch, shuffle=True),
        DataLoader(val_ds, batch_size=batch*2),
        DataLoader(test_ds, batch_size=batch*2)
    )


def train_epoch(model, loader, crit, opt, dev):
    model.train(); total_loss = total_corr = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(dev), yb.to(dev)
        opt.zero_grad()
        logits = model(Xb)
        loss = crit(logits, yb)
        loss.backward(); opt.step()
        total_loss += loss.item() * Xb.size(0)
        total_corr += (logits.argmax(1) == yb).sum().item()
    return total_loss/len(loader.dataset), total_corr/len(loader.dataset)


def eval_model(model, loader, dev):
    model.eval(); corr = 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(dev), yb.to(dev)
            corr += (model(Xb).argmax(1) == yb).sum().item()
    return corr/len(loader.dataset)


def plot_confusion(model, loader, names, dev):
    y_t, y_p = [], []
    model.eval()
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(dev)
            preds = model(Xb).argmax(1).cpu().numpy()
            y_p.extend(preds); y_t.extend(yb.numpy())
    cm = confusion_matrix(y_t, y_p)
    disp = ConfusionMatrixDisplay(cm, display_labels=names)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, cmap="Blues")
    plt.show()


class SimpleCNN1D(nn.Module):
    def __init__(self, in_ch=1, n_cls=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 16, 7, padding=3), nn.ReLU(True), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(True), nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(32, 16), nn.ReLU(True), nn.Linear(16, n_cls)
        )
    def forward(self, x): return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=300, drop=0.1):
        super().__init__(); self.drop = nn.Dropout(drop)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return self.drop(x + self.pe[:, :x.size(1)])


class TransformerClassifier(nn.Module):
    def __init__(self, in_ch=1, d_model=64, heads=4, layers=2, n_cls=4, max_len=300):
        super().__init__()
        self.proj = nn.Linear(in_ch, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        enc = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=128, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.fc = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(True), nn.Linear(128, n_cls))

    def forward(self, x):
        x = x.permute(0,2,1)  # (B,T,C)
        x = self.proj(x)
        x = self.pos(x)
        x = self.enc(x)
        return self.fc(x.mean(1))


def main(model_type='cnn'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patterns = ['histo_PD_exp3_vs_epsilon_v3.npy',
                'histo_PD_exp3_vs_exp3_v3.npy',
                'histo_PD_exp3_vs_ucb_v3.npy',
                'histo_PD_exp3_vs_ftl_v3.npy']
    X, y = load_data(patterns, seq_len=150)
    loaders = get_loaders(X, y, batch=64, seq_len=150)

    model = (SimpleCNN1D(1, len(patterns)) if model_type=='cnn'
             else TransformerClassifier(1, 64, 4, 2, len(patterns), X.shape[1]))
    model.to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    for e in range(1, epochs+1):
        tl, ta = train_epoch(model, loaders[0], crit, opt, device)
        va = eval_model(model, loaders[1], device)
        print(f"{model_type.upper()} Epoch {e:02d}: train_loss={tl:.4f}, train_acc={ta:.3f}, val_acc={va:.3f}")

    te = eval_model(model, loaders[2], device)
    print(f"Test Acc: {te:.3f}")

    class_names = [p.split('_')[-1].split('.')[0] for p in patterns]
    plot_confusion(model, loaders[2], class_names, device)


if __name__ == '__main__':
    main('cnn')  # or main('transformer')
