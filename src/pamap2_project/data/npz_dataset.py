from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset


class NPZWindowsDataset(Dataset):
    """
    Dataset per PAMAP2 già normalizzato.
    Ritorna:
      {'x': Tensor(T,F), 'label': Tensor()}
    """
    def __init__(self, npz_path: str | Path):
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"NPZ non trovato: {self.npz_path}")

        data = np.load(self.npz_path, allow_pickle=True)
        if "X" not in data or "y" not in data:
            raise ValueError(f"{self.npz_path} deve contenere X e y.")

        self.X = data["X"].astype(np.float32)  # (N,T,F)
        self.y = data["y"].astype(np.int64)    # (N,)

        if self.X.ndim != 3:
            raise ValueError(f"Atteso X (N,T,F), ottenuto {self.X.shape}")
        if self.y.ndim != 1:
            raise ValueError(f"Atteso y (N,), ottenuto {self.y.shape}")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("Mismatch tra numero finestre in X e y.")

        # Metadati utili (facoltativi)
        self.feature_cols = data["feature_cols"] if "feature_cols" in data else None
        self.activities = data["activities"] if "activities" in data else None

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = torch.from_numpy(self.X[idx]).float()     # (T,F)
        label = torch.tensor(self.y[idx]).long()      # ()
        return {"x": x, "label": label}
