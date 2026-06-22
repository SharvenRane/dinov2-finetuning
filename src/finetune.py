"""Training utilities: freezing, optimizer construction, and the train loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .head import ClassifierModel


def set_backbone_trainable(model: ClassifierModel, trainable: bool) -> None:
    """Freeze or unfreeze the backbone in place.

    Freezing sets ``requires_grad=False`` on every backbone parameter so the
    optimizer never updates them. The head stays trainable in both cases.
    """
    for p in model.backbone.parameters():
        p.requires_grad = trainable


def trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def build_optimizer(
    model: ClassifierModel,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    backbone_lr: Optional[float] = None,
) -> torch.optim.Optimizer:
    """Build an AdamW optimizer over the currently trainable parameters.

    When ``backbone_lr`` is given and the backbone is trainable, the backbone
    gets its own (usually smaller) learning rate while the head keeps ``lr``.
    This mirrors the common DINOv2 finetuning recipe of a low backbone rate and
    a higher head rate. Frozen parameters are excluded automatically.
    """
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]

    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": lr})
    if backbone_params:
        groups.append(
            {"params": backbone_params, "lr": backbone_lr if backbone_lr else lr}
        )

    if not groups:
        raise ValueError("no trainable parameters; unfreeze the backbone or head")

    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


@dataclass
class TrainHistory:
    """Per epoch metrics collected during training."""

    loss: List[float] = field(default_factory=list)
    accuracy: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, List[float]]:
        return {"loss": self.loss, "accuracy": self.accuracy}


def train_classifier(
    model: ClassifierModel,
    loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    backbone_lr: Optional[float] = None,
    device: str = "cpu",
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> TrainHistory:
    """Train the model on a labelled loader and return the metric history.

    The loop is a standard cross entropy classification fit. Whether the
    backbone updates depends entirely on its ``requires_grad`` state, which the
    caller controls through :func:`set_backbone_trainable`.
    """
    model.to(device)
    if optimizer is None:
        optimizer = build_optimizer(
            model, lr=lr, weight_decay=weight_decay, backbone_lr=backbone_lr
        )
    criterion = nn.CrossEntropyLoss()
    history = TrainHistory()

    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.shape[0]
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.shape[0]

        history.loss.append(running_loss / total)
        history.accuracy.append(correct / total)

    return history
