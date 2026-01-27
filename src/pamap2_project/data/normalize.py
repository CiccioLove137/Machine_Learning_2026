from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


SPLITS_DIR = Path("data/processed/splits")
NORM_DIR = Path("data/processed/normalized")


def _load_split(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}. Hai eseguito lo stage split?")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _compute_train_mean_std(X_train: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    X_train: (N, T, F)
    mean/std per feature, stimati su tutti i campioni temporali e su tutte le finestre.
    """
    if X_train.ndim != 3:
        raise ValueError(f"Atteso X_train 3D (N,T,F), ottenuto {X_train.shape}")

    # (N*T, F)
    X_flat = X_train.reshape(-1, X_train.shape[-1]).astype(np.float32)

    mean = X_flat.mean(axis=0)
    std = X_flat.std(axis=0)

    # evita divisioni per zero (feature costanti)
    std = np.where(std < eps, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def _save_split(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def run_normalize(cfg: dict) -> None:
    NORM_DIR.mkdir(parents=True, exist_ok=True)

    train_path = SPLITS_DIR / "train.npz"
    val_path = SPLITS_DIR / "val.npz"
    test_path = SPLITS_DIR / "test.npz"

    train = _load_split(train_path)
    val = _load_split(val_path)
    test = _load_split(test_path)

    if "X" not in train or "y" not in train:
        raise ValueError("train.npz deve contenere almeno X e y.")
    if "X" not in val or "y" not in val:
        raise ValueError("val.npz deve contenere almeno X e y.")
    if "X" not in test or "y" not in test:
        raise ValueError("test.npz deve contenere almeno X e y.")

    X_train = train["X"].astype(np.float32)
    X_val = val["X"].astype(np.float32)
    X_test = test["X"].astype(np.float32)

    # 1) calcolo mean/std solo sul train
    mean, std = _compute_train_mean_std(X_train)

    # 2) applico normalizzazione
    train["X"] = _apply_norm(X_train, mean, std)
    val["X"] = _apply_norm(X_val, mean, std)
    test["X"] = _apply_norm(X_test, mean, std)

    # 3) salvo anche i parametri di normalizzazione (Per riproducibilità)
    norm_meta = {
        "norm_mean": mean,
        "norm_std": std,
        "norm_method": np.array(["zscore_train_only"], dtype=object),
    }

    # aggiungi meta ai tre split
    train.update(norm_meta)
    val.update(norm_meta)
    test.update(norm_meta)

    # 4) salva
    _save_split(NORM_DIR / "train.npz", train)
    _save_split(NORM_DIR / "val.npz", val)
    _save_split(NORM_DIR / "test.npz", test)

    print("Normalizzazione completata (train-only z-score). Salvati:")
    print(f" - {NORM_DIR / 'train.npz'} | X={train['X'].shape} y={train['y'].shape}")
    print(f" - {NORM_DIR / 'val.npz'}   | X={val['X'].shape} y={val['y'].shape}")
    print(f" - {NORM_DIR / 'test.npz'}  | X={test['X'].shape} y={test['y'].shape}")
    print(f"Mean/std salvati come: norm_mean, norm_std")
