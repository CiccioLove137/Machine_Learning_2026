from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.pamap2_project.data.npz_dataset import NPZWindowsDataset
from src.pamap2_project.models.rnn import RNNClassifier
from src.pamap2_project.utils.metrics import compute_metrics, evaluate

NORM_DIR = Path("data/processed/normalized")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    predictions: List[int] = []
    references: List[int] = []

    for batch in tqdm(dataloader, desc="Training"):
        x = batch["x"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(x)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        pred = torch.argmax(outputs, dim=1)
        predictions.extend(pred.detach().cpu().numpy().tolist())
        references.extend(labels.detach().cpu().numpy().tolist())

    metrics = compute_metrics(predictions, references)
    metrics["loss"] = running_loss / max(1, len(dataloader))
    return metrics


def _get_device(training_cfg: dict) -> torch.device:
    device_pref = str(training_cfg.get("device", "cuda")).lower()
    use_cuda = (device_pref == "cuda") and torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")


def run_train_rnn(cfg: dict, run_dir: Path) -> None:
    # -----------------------
    # CONFIG
    # -----------------------
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]
    dataset_cfg = cfg["dataset"]

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    epochs = int(training_cfg["epochs"])
    batch_size = int(training_cfg["batch_size"])
    lr = float(training_cfg["lr"])
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    optimizer_name = str(training_cfg.get("optimizer", "adam")).lower()

    evaluation_metric = str(training_cfg.get("evaluation_metric", "f1"))
    lower_is_better = bool(training_cfg.get("best_metric_lower_is_better", False))

    # Early stopping
    es_cfg = training_cfg.get("early_stopping", {})
    es_enabled = bool(es_cfg.get("enabled", True))
    es_patience = int(es_cfg.get("patience", 5))
    es_min_delta = float(es_cfg.get("min_delta", 0.0))
    es_monitor = str(es_cfg.get("monitor", evaluation_metric))

    # Scheduler
    sch_cfg = training_cfg.get("scheduler", {})
    sch_enabled = bool(sch_cfg.get("enabled", True))
    sch_name = str(sch_cfg.get("name", "reduce_on_plateau")).lower()
    sch_factor = float(sch_cfg.get("factor", 0.5))
    sch_patience = int(sch_cfg.get("patience", 2))
    sch_min_lr = float(sch_cfg.get("min_lr", 1e-5))
    sch_monitor = str(sch_cfg.get("monitor", es_monitor))

    device = _get_device(training_cfg)
    print(f"Device: {device}")

    # -----------------------
    # LOAD DATA
    # -----------------------
    train_path = NORM_DIR / "train.npz"
    val_path = NORM_DIR / "val.npz"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "NPZ normalizzati non trovati. Esegui prima windows/split/normalize.\n"
            f"- {train_path}\n- {val_path}"
        )

    train_ds = NPZWindowsDataset(train_path)
    val_ds = NPZWindowsDataset(val_path)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    sample_x = train_ds[0]["x"]  # (T,F)
    _, input_dim = sample_x.shape
    num_classes = len(dataset_cfg["activities"])

    # -----------------------
    # MODEL
    # -----------------------
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

    print(
        f"Model: {rnn_type.upper()} | input_dim={input_dim} | hidden_dim={hidden_dim} | "
        f"layers={num_layers} | dropout={dropout} | classes={num_classes}"
    )

    # -----------------------
    # OPTIM / LOSS / SCHED
    # -----------------------
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Ottimizzatore non supportato: {optimizer_name}")

    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    reduce_on_plateau = None

    if sch_enabled:
        if sch_name != "reduce_on_plateau":
            raise ValueError(f"Scheduler non supportato: {sch_name} (usa reduce_on_plateau)")

        # NOTA: niente verbose=True -> evita il warning
        reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=("min" if lower_is_better else "max"),
            factor=sch_factor,
            patience=sch_patience,
            min_lr=sch_min_lr,
        )

    print(
        f"EarlyStopping: enabled={es_enabled} patience={es_patience} min_delta={es_min_delta} monitor={es_monitor} "
        f"| lower_is_better={lower_is_better}"
    )
    print(
        f"Scheduler: enabled={sch_enabled} name={sch_name} factor={sch_factor} patience={sch_patience} min_lr={sch_min_lr} "
        f"| monitor={sch_monitor}"
    )
    print()

    # -----------------------
    # TRAIN LOOP
    # -----------------------
    best_val_metric = np.inf if lower_is_better else -np.inf
    best_state = None

    train_hist: List[Dict[str, float]] = []
    val_hist: List[Dict[str, float]] = []

    # early stopping trackers
    es_best = np.inf if lower_is_better else -np.inf
    es_bad_epochs = 0

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_dl, criterion, optimizer, device)
        val_metrics = evaluate(model, val_dl, criterion, device)

        train_metrics["epoch"] = epoch
        val_metrics["epoch"] = epoch

        train_hist.append(train_metrics)
        val_hist.append(val_metrics)

        if evaluation_metric not in val_metrics:
            raise KeyError(
                f"evaluation_metric='{evaluation_metric}' non presente in val_metrics={list(val_metrics.keys())}"
            )

        # lr value
        current_lr = float(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch:03d}/{epochs} | lr={current_lr:.6f} | "
            f"Train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} f1={train_metrics['f1']:.4f} | "
            f"Val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1']:.4f}"
        )

        # -----------------------
        # Scheduler step (on metric)
        # -----------------------
        if reduce_on_plateau is not None:
            if sch_monitor not in val_metrics:
                raise KeyError(f"scheduler.monitor='{sch_monitor}' non presente in val_metrics={list(val_metrics.keys())}")
            reduce_on_plateau.step(val_metrics[sch_monitor])

        # -----------------------
        # Best model (save in RAM)
        # -----------------------
        current = float(val_metrics[evaluation_metric])
        is_best = (current < best_val_metric) if lower_is_better else (current > best_val_metric)
        if is_best:
            print(f"New best model found with val {evaluation_metric}: {current:.4f}")
            best_val_metric = current
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # -----------------------
        # Early stopping
        # -----------------------
        if es_enabled:
            if es_monitor not in val_metrics:
                raise KeyError(f"early_stopping.monitor='{es_monitor}' non presente in val_metrics={list(val_metrics.keys())}")

            monitored = float(val_metrics[es_monitor])

            improved = (monitored < es_best - es_min_delta) if lower_is_better else (monitored > es_best + es_min_delta)

            if improved:
                es_best = monitored
                es_bad_epochs = 0
            else:
                es_bad_epochs += 1
                if es_bad_epochs >= es_patience:
                    print(f"\n[EarlyStopping] STOP at epoch {epoch} (no improvement on {es_monitor} for {es_patience} epochs)\n")
                    break

    # -----------------------
    # SAVE ARTIFACTS
    # -----------------------
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "train_history.json").write_text(json.dumps(train_hist, indent=2), encoding="utf-8")
    (run_dir / "val_history.json").write_text(json.dumps(val_hist, indent=2), encoding="utf-8")

    if best_state is None:
        best_state = model.state_dict()

    best_path = run_dir / "best_model.pt"
    torch.save(best_state, best_path)
    print(f"Best model saved to: {best_path}")
