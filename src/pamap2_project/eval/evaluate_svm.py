from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import joblib

from src.pamap2_project.utils.metrics import compute_metrics


FEATS_DIR = Path("data/processed/features_svm")


def _load_feats(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}. Hai eseguito stage svm-features?")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def run_evaluate_svm(cfg: dict, run_dir: Path) -> None:
    model_path = run_dir / "svm_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Modello SVM non trovato: {model_path}\nFai prima: --stage train-svm")

    test_path = FEATS_DIR / "test.npz"
    test = _load_feats(test_path)

    if "X_feat" not in test or "y" not in test:
        raise ValueError("features test.npz deve contenere X_feat e y")

    X_test = test["X_feat"].astype(np.float32)
    y_test = test["y"].astype(np.int64)

    model = joblib.load(model_path)

    pred = model.predict(X_test).tolist()
    ref = y_test.tolist()
    test_metrics = compute_metrics(pred, ref)

    print("\n=== TEST METRICS (SVM) ===")
    for k, v in test_metrics.items():
        print(f"Test {k}: {v:.4f}")

    out_path = run_dir / "test_metrics.json"
    out_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
