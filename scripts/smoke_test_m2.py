"""M2 Smoke Test — verify all data pipelines work end-to-end."""

from authentix.pipelines.data_registry import DataRegistry

reg = DataRegistry("d:/Files/DMCE/Authentix/data")

# 1. Check availability
print("=== Available Datasets ===")
for name, avail in reg.list_available().items():
    print(f"  {name}: {'YES' if avail else 'NO'}")

# 2. CIFAKE
print("\n=== CIFAKE Smoke Test ===")
loader = reg.get_image_dataloader(
    "cifake", split="train", batch_size=4, max_samples_per_class=8
)
batch = next(iter(loader))
print(f"  Batch images shape: {batch['image'].shape}")
print(f"  Batch labels: {batch['label'].tolist()}")

# 3. GenImage
print("\n=== GenImage Smoke Test ===")
loader2 = reg.get_image_dataloader(
    "genimage", split="train", batch_size=4, max_samples_per_class=4
)
batch2 = next(iter(loader2))
print(f"  Batch images shape: {batch2['image'].shape}")
print(f"  Batch labels: {batch2['label'].tolist()}")
print(f"  Generators: {batch2['generator']}")

# 4. URL features
print("\n=== URL Feature Smoke Test ===")
url_ds = reg.get_url_dataset(max_samples=100)
X, y = url_ds.get_features_and_labels()
print(f"  Feature matrix shape: {X.shape}")
print(f"  Label distribution: benign={sum(y==0)}, malicious={sum(y==1)}")

print("\n=== ALL M2 SMOKE TESTS PASSED ===")
