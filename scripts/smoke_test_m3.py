"""M3 Smoke Test — verify model architecture and training loop work."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from authentix.models.image_classifier import ImageClassifier
from authentix.pipelines.data_registry import DataRegistry
from authentix.pipelines.image.transforms import get_train_transforms

print("=== Model Architecture Test ===")
model = ImageClassifier(
    num_classes=1,
    dropout=0.4,
    freeze_backbone=True,
    unfreeze_last_n_blocks=2,
    pretrained=True,
)

# Test forward pass with dummy input
dummy_input = torch.randn(2, 3, 380, 380)
logits = model(dummy_input)
print(f"  Input shape:  {dummy_input.shape}")
print(f"  Output shape: {logits.shape}")
print(f"  Output values: {logits.squeeze().tolist()}")

# Test predict_proba
probs = model.predict_proba(dummy_input)
print(f"  Probabilities: {probs.tolist()}")
assert all(0 <= p <= 1 for p in probs.tolist()), "Probabilities out of range!"

# Test feature maps for Grad-CAM
fmaps = model.get_feature_maps(dummy_input)
print(f"  Feature maps shape: {fmaps.shape}")

# Test with real data
print("\n=== Real Data Forward Pass ===")
registry = DataRegistry("d:/Files/DMCE/Authentix/data")
loader = registry.get_image_dataloader(
    "cifake", split="train", batch_size=2, max_samples_per_class=2
)
batch = next(iter(loader))
logits = model(batch["image"])
probs = torch.sigmoid(logits).squeeze()
print(f"  Real batch labels: {batch['label'].tolist()}")
print(f"  Model predictions: {probs.tolist()}")

print("\n=== ALL M3 SMOKE TESTS PASSED ===")
