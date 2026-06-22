"""Backbone definitions for the finetuning harness.

The harness is written against a small interface so that tests can run with a
tiny randomly initialised Vision Transformer instead of downloading the real
DINOv2 weights. Any module that exposes an ``embed_dim`` attribute and maps a
batch of images ``[B, 3, H, W]`` to pooled features ``[B, embed_dim]`` works as
a backbone here. The real DINOv2 backbones from timm satisfy the same contract,
so the training and evaluation code is identical whether you run the tiny test
model or a downloaded checkpoint.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Split an image into patches and project each patch to a token."""

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            raise ValueError(
                f"expected input of size {self.img_size}, got {tuple(x.shape[-2:])}"
            )
        x = self.proj(x)  # [B, embed_dim, grid, grid]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class TinyViTBackbone(nn.Module):
    """A small ViT encoder that mirrors the DINOv2 output contract.

    It produces a pooled embedding per image. Pooling uses the CLS token, which
    matches how the real DINOv2 ``forward`` returns its class token feature.
    The model is intentionally tiny so it trains in milliseconds on CPU.
    """

    def __init__(
        self,
        img_size: int = 16,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 32,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            encoder_layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        tokens = self.patch_embed(x)  # [B, N, D]
        cls = self.cls_token.expand(b, -1, -1)  # [B, 1, D]
        tokens = torch.cat([cls, tokens], dim=1)  # [B, N+1, D]
        tokens = tokens + self.pos_embed
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)
        return tokens[:, 0]  # pooled CLS feature [B, D]


class _TimmWrapper(nn.Module):
    """Adapt a timm DINOv2 model to the backbone contract used here."""

    def __init__(self, model: nn.Module, embed_dim: int):
        super().__init__()
        self.model = model
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model(x)
        if feats.dim() != 2:
            feats = feats.reshape(feats.shape[0], -1)
        return feats


def load_backbone(
    name: str = "tiny",
    pretrained: bool = False,
    img_size: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """Return a backbone module.

    ``name="tiny"`` builds the local :class:`TinyViTBackbone` and never touches
    the network, which is what the tests use. Any other name is treated as a
    timm model id (for example ``vit_small_patch14_dinov2.lvd142m``) and is
    created through timm with global pooling enabled so it returns a single
    feature vector per image.
    """
    if name == "tiny":
        if img_size is not None:
            kwargs["img_size"] = img_size
        return TinyViTBackbone(**kwargs)

    import timm  # imported lazily so the tiny path has no timm dependency

    model = timm.create_model(name, pretrained=pretrained, num_classes=0, **kwargs)
    embed_dim = getattr(model, "num_features", None)
    if embed_dim is None:
        with torch.no_grad():
            size = img_size or 224
            probe = torch.zeros(1, 3, size, size)
            embed_dim = model(probe).shape[-1]
    return _TimmWrapper(model, int(embed_dim))
