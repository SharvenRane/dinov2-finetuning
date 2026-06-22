"""Evaluation of a trained classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class EvalResult:
    """Held metrics from one evaluation pass."""

    loss: float
    accuracy: float
    num_samples: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "num_samples": float(self.num_samples),
        }


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> EvalResult:
    """Run the model over a loader and report mean loss and accuracy."""
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.shape[0]

    if total == 0:
        raise ValueError("evaluation loader produced no samples")

    return EvalResult(
        loss=total_loss / total,
        accuracy=correct / total,
        num_samples=total,
    )
