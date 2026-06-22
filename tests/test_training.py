import torch
from torch.utils.data import DataLoader

from src.backbone import TinyViTBackbone
from src.data import FeatureDataset, make_separable_features
from src.evaluate import evaluate_classifier
from src.finetune import set_backbone_trainable, train_classifier
from src.head import ClassifierModel


def _split_loaders(seed=0):
    x, y = make_separable_features(
        num_per_class=40, num_classes=3, img_size=16, seed=seed
    )
    n = x.shape[0]
    cut = int(n * 0.75)
    train = FeatureDataset(x[:cut], y[:cut])
    test = FeatureDataset(x[cut:], y[cut:])
    return (
        DataLoader(train, batch_size=16, shuffle=True),
        DataLoader(test, batch_size=16, shuffle=False),
    )


def _build_model():
    bb = TinyViTBackbone(img_size=16, patch_size=4, embed_dim=32)
    return ClassifierModel(bb, num_classes=3)


def test_linear_head_loss_decreases():
    torch.manual_seed(0)
    model = _build_model()
    set_backbone_trainable(model, False)
    train_loader, _ = _split_loaders()
    history = train_classifier(model, train_loader, epochs=15, lr=1e-2)
    assert history.loss[-1] < history.loss[0], (
        f"loss did not decrease: {history.loss[0]:.4f} -> {history.loss[-1]:.4f}"
    )


def test_linear_head_improves_accuracy_on_separable_data():
    torch.manual_seed(0)
    model = _build_model()
    set_backbone_trainable(model, False)
    train_loader, test_loader = _split_loaders()

    before = evaluate_classifier(model, test_loader)
    train_classifier(model, train_loader, epochs=30, lr=1e-2)
    after = evaluate_classifier(model, test_loader)

    assert after.accuracy > before.accuracy
    assert after.accuracy >= 0.75, f"final accuracy too low: {after.accuracy:.3f}"


def test_finetuning_full_model_also_learns():
    torch.manual_seed(0)
    model = _build_model()
    set_backbone_trainable(model, True)
    train_loader, test_loader = _split_loaders()

    before = evaluate_classifier(model, test_loader)
    train_classifier(model, train_loader, epochs=20, lr=5e-3, backbone_lr=5e-4)
    after = evaluate_classifier(model, test_loader)

    assert after.accuracy > before.accuracy


def test_eval_result_reports_sample_count():
    model = _build_model()
    _, test_loader = _split_loaders()
    result = evaluate_classifier(model, test_loader)
    assert result.num_samples == len(test_loader.dataset)
    assert 0.0 <= result.accuracy <= 1.0
