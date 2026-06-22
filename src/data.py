"""Synthetic data helpers for the harness tests and quick experiments.

The harness expects image batches, but for fast CPU testing we want data whose
class structure is easy to learn so a linear head can clearly separate it. The
generator below paints a low frequency coloured pattern onto each image whose
parameters depend on the class label. A randomly initialised backbone passes
that structure through, and a linear head trained on the pooled features can
recover the class. This gives a property we can assert: training the head lowers
the loss and raises accuracy.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch.utils.data import Dataset


def make_separable_features(
    num_per_class: int,
    num_classes: int,
    img_size: int = 16,
    in_chans: int = 3,
    noise: float = 0.15,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a synthetic image classification set with separable structure.

    Each class gets its own spatial gradient direction and channel offset, so
    the resulting images form well separated clusters once a backbone embeds
    them. Returns ``(images, labels)`` with images shaped
    ``[N, in_chans, img_size, img_size]``.
    """
    if num_per_class <= 0 or num_classes <= 0:
        raise ValueError("num_per_class and num_classes must be positive")

    g = torch.Generator().manual_seed(seed)
    coords = torch.linspace(-1.0, 1.0, img_size)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")

    images = []
    labels = []
    for c in range(num_classes):
        angle = (c / num_classes) * 2.0 * math.pi
        # A class specific oriented plane wave gives a clean per class signature.
        base = math.cos(angle) * xx + math.sin(angle) * yy
        pattern = torch.stack(
            [base * (1.0 + 0.3 * ch) + 0.2 * ch for ch in range(in_chans)], dim=0
        )  # [C, H, W]
        for _ in range(num_per_class):
            noise_tensor = noise * torch.randn(
                in_chans, img_size, img_size, generator=g
            )
            images.append(pattern + noise_tensor)
            labels.append(c)

    x = torch.stack(images, dim=0)
    y = torch.tensor(labels, dtype=torch.long)

    perm = torch.randperm(x.shape[0], generator=g)
    return x[perm], y[perm]


class FeatureDataset(Dataset):
    """Wrap image and label tensors as a torch Dataset."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        if images.shape[0] != labels.shape[0]:
            raise ValueError("images and labels must have the same length")
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]
