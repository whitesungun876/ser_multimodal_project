# src/utils/dataset_loader.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

LABEL2ID = {"ang": 0, "hap": 1, "sad": 2, "neu": 3}

class EarlyFusionDataset(Dataset):
    def __init__(self, x_path, y_path):
        if x_path.endswith(".npz"):
            data = np.load(x_path)
            self.X = data["X"].astype(np.float32)
            self.y = data["y"]
        else:
            self.X = np.load(x_path).astype(np.float32)
            self.y = np.load(y_path)

        assert len(self.X) == len(self.y), "Mismatch between number of samples and labels"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        if x.ndim == 2:
            # (H, W) -> (1, H, W)
            x = x.unsqueeze(0)
        y_raw = self.y[idx]
        if isinstance(y_raw, (str, np.str_)):
            label = LABEL2ID[y_raw]
        else:
            label = int(y_raw)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

def get_dataloaders(x_path, y_path, batch_size=32, val_split=0.2, shuffle=True):
    dataset = EarlyFusionDataset(x_path, y_path)
    total = len(dataset)
    val_n = int(total * val_split)
    train_n = total - val_n

    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_n, val_n])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
