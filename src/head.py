"""Classification head and the full backbone plus head model."""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    """A single linear layer mapping pooled features to class logits."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        if in_features <= 0 or num_classes <= 0:
            raise ValueError("in_features and num_classes must be positive")
        self.fc = nn.Linear(in_features, num_classes)
        nn.init.trunc_normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ClassifierModel(nn.Module):
    """Backbone plus a linear head.

    The backbone produces a pooled embedding and the head turns it into logits.
    When the backbone is frozen we still run it in the forward pass, but its
    parameters carry no gradient so only the head learns.
    """

    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        embed_dim = getattr(backbone, "embed_dim", None)
        if embed_dim is None:
            raise AttributeError("backbone must expose an 'embed_dim' attribute")
        self.backbone = backbone
        self.head = LinearHead(embed_dim, num_classes)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
