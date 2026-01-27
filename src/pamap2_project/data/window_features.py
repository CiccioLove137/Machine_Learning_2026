from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

NORM_DIR = Path("data/processed/normalized")
FEATS_DIR = Path("data/processed/features_svm")


def _load_npz(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _save_npz(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _stats_features(X: np.ndarray, stats: List[str]) -> np.ndarray:
    """
    X: (N, T, F) -> X_feat: (N, F * len(stats))
    """
    if X.ndim != 3:
        raise ValueError(f"Atteso X 3D (N,T,F), ottenuto {X.shape}")

    parts: List[np.ndarray] = []

    if "mean" in stats:
        parts.append(X.mean(axis=1))
    if "std" in stats:
        parts.append(X.std(axis=1))
    if "min" in stats:
        parts.append(X.min(axis=1))
    if "max" in stats:
        parts.append(X.max(axis=1))

    if not parts:
        raise ValueError(f"Nessuna statistica valida in stats={stats}")

    return np.concatenate(parts, axis=1).astype(np.float32)


def _flatten_features(X: np.ndarray) -> np.ndarray:
    """
    X: (N, T, F) -> (N, T*F)
    """
    if X.ndim != 3:
        raise ValueError(f"Atteso X 3D (N,T,F), ottenuto {X.shape}")
    return X.reshape(X.shape[0], -1).astype(np.float32)


def _build_features(X: np.ndarray, mode: str, stats: List[str]) -> np.ndarray:
    mode = mode.lower()
    if mode == "stats":
        return _stats_features(X, stats)
    if mode == "flatten":
        return _flatten_features(X)
    raise ValueError(f"feature_mode non supportata: {mode} (usa stats|flatten)")


def run_make_svm_features(cfg: dict) -> None:
    svm_cfg = cfg.get("svm", {})
    mode = str(svm_cfg.get("feature_mode", "stats")).lower()
    stats = [str(s).lower() for s in svm_cfg.get("stats", ["mean", "std", "min", "max"])]

    FEATS_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        in_path = NORM_DIR / f"{split_name}.npz"
        data = _load_npz(in_path)

        if "X" not in data or "y" not in data:
            raise ValueError(f"{in_path} deve contenere X e y")

        X = data["X"].astype(np.float32)   # (N,T,F)
        y = data["y"].astype(np.int64)     # (N,)

        X_feat = _build_features(X, mode=mode, stats=stats)

        out = dict(data)
        out["X_feat"] = X_feat
        out["feature_mode"] = np.array([mode], dtype=object)
        out["feature_stats"] = np.array(stats, dtype=object)

        out_path = FEATS_DIR / f"{split_name}.npz"
        _save_npz(out_path, out)

        print(f"[svm-features] {split_name}: X={X.shape} -> X_feat={X_feat.shape} | saved: {out_path}")

    print("\nDone. Feature SVM salvate in data/processed/features_svm/")
