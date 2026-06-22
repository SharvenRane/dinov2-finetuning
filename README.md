# dinov2-finetuning

A small, tested harness for adapting a DINOv2 style self supervised backbone to a
new labelled task. It covers the three steps you actually do when you take a
frozen self supervised encoder into a downstream domain: decide whether to freeze
or finetune the backbone, attach a linear classification head on top of the pooled
features, train, and evaluate.

The whole thing is written against a tiny backbone interface so the tests run on
CPU in a couple of seconds with no weight downloads. The same training and
evaluation code runs unchanged against a real DINOv2 checkpoint from timm.

## Why a tiny backbone

The real DINOv2 weights are large and pulling them on every test run is slow and
flaky. The harness only assumes a backbone that exposes an `embed_dim` attribute
and maps an image batch `[B, 3, H, W]` to pooled features `[B, embed_dim]`. The
timm DINOv2 models satisfy that contract, and so does the `TinyViTBackbone` in
`src/backbone.py`, which is a randomly initialised small Vision Transformer with a
CLS token. Tests use the tiny model. Production code points `load_backbone` at a
timm model id instead.

## What is inside

```
src/
  backbone.py   TinyViTBackbone plus load_backbone for the timm DINOv2 path
  head.py       LinearHead and the ClassifierModel that joins backbone and head
  data.py       synthetic separable image generator and a Dataset wrapper
  finetune.py   freeze and unfreeze, optimizer groups, the training loop
  evaluate.py   loss and accuracy evaluation
tests/          pytest behaviour checks
```

## Freeze versus finetune

`set_backbone_trainable(model, trainable)` flips `requires_grad` on the backbone
parameters. When the backbone is frozen the optimizer never receives those tensors,
so only the head learns and the backbone weights come out of training byte for
byte identical. When the backbone is trainable, `build_optimizer` can give it a
separate, usually smaller, learning rate while the head keeps the larger one. That
matches the common recipe of nudging the backbone gently and letting the head move
faster.

## Quick start

```python
from torch.utils.data import DataLoader
from src.backbone import load_backbone
from src.head import ClassifierModel
from src.data import make_separable_features, FeatureDataset
from src.finetune import set_backbone_trainable, train_classifier
from src.evaluate import evaluate_classifier

x, y = make_separable_features(num_per_class=40, num_classes=3, img_size=16)
loader = DataLoader(FeatureDataset(x, y), batch_size=16, shuffle=True)

backbone = load_backbone("tiny", img_size=16, patch_size=4, embed_dim=32)
model = ClassifierModel(backbone, num_classes=3)

set_backbone_trainable(model, False)          # linear probe on a frozen backbone
train_classifier(model, loader, epochs=30, lr=1e-2)
print(evaluate_classifier(model, loader).accuracy)
```

To use a real DINOv2 backbone instead, swap the first line for something like

```python
backbone = load_backbone("vit_small_patch14_dinov2.lvd142m", pretrained=True)
```

and feed it 224 pixel (or the model's native) inputs. Everything downstream stays
the same.

## Tests

```
python -m pytest tests/ -q
```

The suite checks behaviour rather than fixed numbers:

- the backbone returns pooled features of the right shape and exposes `embed_dim`
- freezing leaves every backbone weight unchanged after a training run
- finetuning does change backbone weights
- the optimizer drops frozen parameters and keeps the head
- on synthetic separable data the linear head lowers its loss and raises accuracy
- the synthetic generator is balanced across classes and reproducible by seed

On the last local run all 17 tests passed in about two seconds on CPU.

## Notes

The synthetic data paints a class specific oriented pattern onto each image so the
embedded features form clean clusters. That gives the accuracy tests a real signal
to learn without standing in for any benchmark. No benchmark numbers are claimed
here; the only figures quoted come from the run above.
