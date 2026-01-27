from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_metrics(predictions: List[int], references: List[int]) -> Dict[str, float]:
    acc = accuracy_score(references, predictions)
    precision = precision_score(references, predictions, average="macro", zero_division=0)
    recall = recall_score(references, predictions, average="macro", zero_division=0)
    f1 = f1_score(references, predictions, average="macro", zero_division=0)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


@torch.no_grad()
def evaluate(model, dataloader, criterion, device) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    predictions: List[int] = []
    references: List[int] = []

    for batch in dataloader:
        x = batch["x"].to(device)
        labels = batch["label"].to(device)

        outputs = model(x)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        pred = torch.argmax(outputs, dim=1)
        predictions.extend(pred.cpu().numpy().tolist())
        references.extend(labels.cpu().numpy().tolist())

    metrics = compute_metrics(predictions, references)
    metrics["loss"] = running_loss / max(1, len(dataloader))
    return metrics
