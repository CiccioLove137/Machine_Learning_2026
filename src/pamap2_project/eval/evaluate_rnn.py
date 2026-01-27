import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.pamap2_project.data.npz_dataset import NPZWindowsDataset
from src.pamap2_project.models.rnn import RNNClassifier
from src.pamap2_project.utils.metrics import evaluate


NORM_DIR = Path("data/processed/normalized")


def run_evaluate_rnn(cfg: dict, run_dir: Path) -> None:
    training = cfg["training"]
    model_cfg = cfg["model"]
    dataset_cfg = cfg["dataset"]

    batch_size = int(training["batch_size"])

    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint non trovato nella run: {ckpt_path}\n"
            f"Fai prima: --stage train-rnn"
        )

    device_pref = str(training.get("device", "cuda")).lower()
    device = torch.device("cuda" if (device_pref == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    test_path = NORM_DIR / "test.npz"
    if not test_path.exists():
        raise FileNotFoundError(
            f"NPZ test normalizzato non trovato: {test_path}\n"
            "Esegui prima: --stage normalize"
        )

    test_ds = NPZWindowsDataset(test_path)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    sample_x = test_ds[0]["x"]  # (T,F)
    _, input_dim = sample_x.shape
    num_classes = len(dataset_cfg["activities"])

    rnn_type = str(model_cfg["name"]).lower()
    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    num_layers = int(model_cfg.get("num_layers", 2))
    dropout = float(model_cfg.get("dropout", 0.0))
    bidirectional = bool(model_cfg.get("bidirectional", False))

    model = RNNClassifier(
        rnn_type=rnn_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device)

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    print(f"Model loaded from: {ckpt_path}")

    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate(model, test_dl, criterion, device)

    print("\n=== TEST METRICS ===")
    for k, v in test_metrics.items():
        print(f"Test {k}: {v:.4f}")

    out_path = run_dir / "test_metrics.json"
    out_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
