import copy

import torch
from torch.utils.data import DataLoader

from src.backbone import TinyViTBackbone
from src.data import FeatureDataset, make_separable_features
from src.finetune import (
    build_optimizer,
    set_backbone_trainable,
    train_classifier,
)
from src.head import ClassifierModel


def _make_loader(seed=0):
    x, y = make_separable_features(
        num_per_class=12, num_classes=3, img_size=16, seed=seed
    )
    return DataLoader(FeatureDataset(x, y), batch_size=9, shuffle=True)


def _build_model():
    bb = TinyViTBackbone(img_size=16, patch_size=4, embed_dim=32)
    return ClassifierModel(bb, num_classes=3)


def _backbone_snapshot(model):
    return {k: v.detach().clone() for k, v in model.backbone.state_dict().items()}


def test_freezing_sets_requires_grad_false():
    model = _build_model()
    set_backbone_trainable(model, False)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())


def test_frozen_backbone_weights_unchanged_after_training():
    model = _build_model()
    set_backbone_trainable(model, False)
    before = _backbone_snapshot(model)

    loader = _make_loader()
    train_classifier(model, loader, epochs=4, lr=1e-2)

    after = model.backbone.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k]), f"backbone param {k} changed"


def test_finetuning_updates_backbone_weights():
    model = _build_model()
    set_backbone_trainable(model, True)
    before = _backbone_snapshot(model)

    loader = _make_loader()
    train_classifier(model, loader, epochs=4, lr=1e-2)

    after = model.backbone.state_dict()
    changed = [
        k
        for k in before
        if before[k].is_floating_point() and not torch.equal(before[k], after[k])
    ]
    assert len(changed) > 0, "no backbone parameters were updated during finetuning"


def test_optimizer_excludes_frozen_backbone():
    model = _build_model()
    set_backbone_trainable(model, False)
    opt = build_optimizer(model, lr=1e-3)
    opt_param_ids = {id(p) for group in opt.param_groups for p in group["params"]}
    for p in model.backbone.parameters():
        assert id(p) not in opt_param_ids
    for p in model.head.parameters():
        assert id(p) in opt_param_ids


def test_head_always_trains_even_when_backbone_frozen():
    model = _build_model()
    set_backbone_trainable(model, False)
    head_before = copy.deepcopy(model.head.state_dict())

    loader = _make_loader()
    train_classifier(model, loader, epochs=4, lr=1e-2)

    head_after = model.head.state_dict()
    changed = any(
        not torch.equal(head_before[k], head_after[k]) for k in head_before
    )
    assert changed, "head should update while the backbone is frozen"
