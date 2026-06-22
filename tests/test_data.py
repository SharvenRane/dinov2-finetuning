import torch

from src.data import FeatureDataset, make_separable_features


def test_separable_features_shapes_and_balance():
    x, y = make_separable_features(num_per_class=10, num_classes=4, img_size=16)
    assert x.shape == (40, 3, 16, 16)
    assert y.shape == (40,)
    counts = torch.bincount(y, minlength=4)
    assert torch.all(counts == 10)


def test_separable_features_are_reproducible():
    x1, y1 = make_separable_features(5, 3, seed=7)
    x2, y2 = make_separable_features(5, 3, seed=7)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_feature_dataset_indexing():
    x, y = make_separable_features(3, 2, img_size=16)
    ds = FeatureDataset(x, y)
    assert len(ds) == 6
    img, label = ds[0]
    assert img.shape == (3, 16, 16)
    assert label.shape == ()


def test_feature_dataset_length_mismatch_raises():
    x = torch.randn(4, 3, 16, 16)
    y = torch.zeros(3, dtype=torch.long)
    try:
        FeatureDataset(x, y)
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched lengths")
