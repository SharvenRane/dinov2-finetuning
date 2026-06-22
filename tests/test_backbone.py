import torch

from src.backbone import TinyViTBackbone, load_backbone


def test_backbone_output_shape():
    bb = TinyViTBackbone(img_size=16, patch_size=4, embed_dim=32)
    x = torch.randn(5, 3, 16, 16)
    out = bb(x)
    assert out.shape == (5, 32)
    assert out.dtype == torch.float32


def test_backbone_exposes_embed_dim():
    bb = load_backbone("tiny", embed_dim=24, img_size=16, patch_size=4)
    assert bb.embed_dim == 24
    out = bb(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 24)


def test_backbone_rejects_wrong_input_size():
    bb = TinyViTBackbone(img_size=16, patch_size=4)
    try:
        bb(torch.randn(1, 3, 20, 20))
    except ValueError:
        return
    raise AssertionError("expected ValueError on wrong input size")


def test_backbone_is_deterministic_in_eval():
    bb = TinyViTBackbone(img_size=16, patch_size=4, embed_dim=16)
    bb.eval()
    x = torch.randn(3, 3, 16, 16)
    with torch.no_grad():
        a = bb(x)
        b = bb(x)
    assert torch.allclose(a, b)
