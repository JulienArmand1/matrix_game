# model.py
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

################################################################################
# Dataclass
################################################################################
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


################################################################################
# Transformer
################################################################################

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
        x = self.input_proj(x)          # (B, T, d_model)
        x = self.pos_encoder(x)         # (B, T, d_model)
        x = self.transformer_encoder(x) # (B, T, d_model)
        x = x.mean(dim=1)               # (B, d_model)
        return self.classifier(x)       # (B, num_classes)
    
################################################################################
# CNN
################################################################################
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
