from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
import joblib

from src.pamap2_project.utils.metrics import compute_metrics


FEATS_DIR = Path("data/processed/features_svm")


def _load_feats(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}. Hai eseguito stage svm-features?")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _make_svm_model(cfg: dict) -> Pipeline:
    svm_cfg = cfg.get("svm", {})

    use_scaler = bool(svm_cfg.get("use_scaler", True))
    kernel = str(svm_cfg.get("kernel", "rbf")).lower()
    C = float(svm_cfg.get("C", 1.0))
    gamma = str(svm_cfg.get("gamma", "scale"))
    cw = svm_cfg.get("class_weight", "balanced")
    class_weight = None if (cw is None or str(cw).lower() == "null") else cw

    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))

    if kernel == "linear":
        # LinearSVC è spesso più veloce
        clf = LinearSVC(C=C, class_weight=class_weight, max_iter=20000)
        steps.append(("svm", clf))
    elif kernel == "rbf":
        clf = SVC(C=C, kernel="rbf", gamma=gamma, class_weight=class_weight)
        steps.append(("svm", clf))
    else:
        raise ValueError(f"kernel non supportato: {kernel} (usa rbf|linear)")

    return Pipeline(steps)


def _eval_on_split(model: Pipeline, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    pred = model.predict(X).tolist()
    ref = y.tolist()
    metrics = compute_metrics(pred, ref)
    # per coerenza con RNN lasciamo loss fuori (SVM non ha loss “uguale”)
    return metrics


def run_train_svm(cfg: dict, run_dir: Path) -> None:
    train_path = FEATS_DIR / "train.npz"
    val_path = FEATS_DIR / "val.npz"

    train = _load_feats(train_path)
    val = _load_feats(val_path)

    if "X_feat" not in train or "y" not in train:
        raise ValueError("features train.npz deve contenere X_feat e y")
    if "X_feat" not in val or "y" not in val:
        raise ValueError("features val.npz deve contenere X_feat e y")

    X_train = train["X_feat"].astype(np.float32)
    y_train = train["y"].astype(np.int64)

    X_val = val["X_feat"].astype(np.float32)
    y_val = val["y"].astype(np.int64)

    model = _make_svm_model(cfg)

    print(f"[SVM] Training on X_train={X_train.shape}, y_train={y_train.shape}")
    model.fit(X_train, y_train)

    train_metrics = _eval_on_split(model, X_train, y_train)
    val_metrics = _eval_on_split(model, X_val, y_val)

    print("\n=== TRAIN METRICS (SVM) ===")
    for k, v in train_metrics.items():
        print(f"Train {k}: {v:.4f}")

    print("\n=== VAL METRICS (SVM) ===")
    for k, v in val_metrics.items():
        print(f"Val {k}: {v:.4f}")

    # salva artefatti nella run
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "svm_model.joblib"
    joblib.dump(model, model_path)

    (run_dir / "train_metrics.json").write_text(json.dumps(train_metrics, indent=2), encoding="utf-8")
    (run_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")

    print(f"\nSaved model: {model_path}")
    print(f"Saved: {run_dir / 'train_metrics.json'}")
    print(f"Saved: {run_dir / 'val_metrics.json'}")
