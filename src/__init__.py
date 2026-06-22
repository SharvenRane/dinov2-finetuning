"""DINOv2 style finetuning and evaluation harness."""

from .backbone import TinyViTBackbone, load_backbone
from .head import LinearHead, ClassifierModel
from .finetune import set_backbone_trainable, build_optimizer, train_classifier
from .evaluate import evaluate_classifier, EvalResult
from .data import make_separable_features, FeatureDataset

__all__ = [
    "TinyViTBackbone",
    "load_backbone",
    "LinearHead",
    "ClassifierModel",
    "set_backbone_trainable",
    "build_optimizer",
    "train_classifier",
    "evaluate_classifier",
    "EvalResult",
    "make_separable_features",
    "FeatureDataset",
]
