"""M2 Gate Tests — Data pipelines and feature extraction."""

import pytest
from pathlib import Path

from authentix.pipelines.image.datasets import CIFAKEDataset, TinyGenImageDataset
from authentix.pipelines.image.transforms import (
    get_train_transforms,
    get_val_transforms,
    denormalise,
)
from authentix.pipelines.url.feature_extractor import (
    extract_url_features,
    extract_url_feature_vector,
    NUM_FEATURES,
    FEATURE_NAMES,
)

# ── Paths ──
DATA_DIR = Path("d:/Files/DMCE/Authentix/data")
CIFAKE_DIR = DATA_DIR / "CIFAKE"
GENIMAGE_DIR = DATA_DIR / "tiny_genimage"


# ═══════════════════════════════════════════════════════════════
# CIFAKE Dataset Tests
# ═══════════════════════════════════════════════════════════════

class TestCIFAKEDataset:
    """Test CIFAKE data loading."""

    @pytest.mark.skipif(
        not CIFAKE_DIR.exists(), reason="CIFAKE data not available"
    )
    def test_load_train(self):
        ds = CIFAKEDataset(CIFAKE_DIR, split="train", max_samples_per_class=10)
        assert len(ds) == 20  # 10 REAL + 10 FAKE
        sample = ds[0]
        assert "image" in sample
        assert "label" in sample
        assert sample["label"] in (0, 1)

    @pytest.mark.skipif(
        not CIFAKE_DIR.exists(), reason="CIFAKE data not available"
    )
    def test_load_test(self):
        ds = CIFAKEDataset(CIFAKE_DIR, split="test", max_samples_per_class=5)
        assert len(ds) == 10

    @pytest.mark.skipif(
        not CIFAKE_DIR.exists(), reason="CIFAKE data not available"
    )
    def test_with_transforms(self):
        transform = get_train_transforms(size=224)
        ds = CIFAKEDataset(
            CIFAKE_DIR, split="train",
            transform=transform, max_samples_per_class=5
        )
        sample = ds[0]
        # Should be a tensor after transforms
        import torch
        assert isinstance(sample["image"], torch.Tensor)
        assert sample["image"].shape == (3, 224, 224)

    @pytest.mark.skipif(
        not CIFAKE_DIR.exists(), reason="CIFAKE data not available"
    )
    def test_class_weights(self):
        ds = CIFAKEDataset(CIFAKE_DIR, split="train", max_samples_per_class=10)
        weights = ds.class_weights
        assert len(weights) == 2
        assert all(w > 0 for w in weights)

    def test_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            CIFAKEDataset("/nonexistent/path", split="train")


# ═══════════════════════════════════════════════════════════════
# Tiny GenImage Dataset Tests
# ═══════════════════════════════════════════════════════════════

class TestTinyGenImageDataset:
    """Test Tiny GenImage data loading."""

    @pytest.mark.skipif(
        not GENIMAGE_DIR.exists(), reason="GenImage data not available"
    )
    def test_load_train_all_generators(self):
        ds = TinyGenImageDataset(
            GENIMAGE_DIR, split="train", max_samples_per_class=5
        )
        assert len(ds) > 0
        sample = ds[0]
        assert "image" in sample
        assert "label" in sample
        assert "generator" in sample

    @pytest.mark.skipif(
        not GENIMAGE_DIR.exists(), reason="GenImage data not available"
    )
    def test_load_single_generator(self):
        ds = TinyGenImageDataset(
            GENIMAGE_DIR,
            split="train",
            generators=["imagenet_ai_0419_biggan"],
            max_samples_per_class=5,
        )
        assert len(ds) == 10  # 5 ai + 5 nature
        for i in range(len(ds)):
            assert ds[i]["generator"] == "imagenet_ai_0419_biggan"

    @pytest.mark.skipif(
        not GENIMAGE_DIR.exists(), reason="GenImage data not available"
    )
    def test_cross_generator_split(self):
        """Simulate cross-generator evaluation: train on BigGAN, val on ADM."""
        train_ds = TinyGenImageDataset(
            GENIMAGE_DIR,
            split="train",
            generators=["imagenet_ai_0419_biggan"],
            max_samples_per_class=5,
        )
        val_ds = TinyGenImageDataset(
            GENIMAGE_DIR,
            split="val",
            generators=["imagenet_ai_0508_adm"],
            max_samples_per_class=5,
        )
        # No overlap in generators
        train_gens = {s["generator"] for s in [train_ds[i] for i in range(len(train_ds))]}
        val_gens = {s["generator"] for s in [val_ds[i] for i in range(len(val_ds))]}
        assert train_gens.isdisjoint(val_gens)

    @pytest.mark.skipif(
        not GENIMAGE_DIR.exists(), reason="GenImage data not available"
    )
    def test_generator_distribution(self):
        ds = TinyGenImageDataset(
            GENIMAGE_DIR, split="train", max_samples_per_class=3
        )
        dist = ds.get_generator_distribution()
        assert isinstance(dist, dict)
        assert len(dist) > 0

    def test_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            TinyGenImageDataset("/nonexistent/path", split="train")


# ═══════════════════════════════════════════════════════════════
# Image Transforms Tests
# ═══════════════════════════════════════════════════════════════

class TestTransforms:
    """Test transform consistency."""

    def test_train_transform_output_shape(self):
        from PIL import Image
        import torch

        transform = get_train_transforms(size=380)
        img = Image.new("RGB", (640, 480), color="blue")
        tensor = transform(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 380, 380)

    def test_val_transform_output_shape(self):
        from PIL import Image
        import torch

        transform = get_val_transforms(size=380)
        img = Image.new("RGB", (100, 100), color="green")
        tensor = transform(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 380, 380)

    def test_denormalise_roundtrip(self):
        import torch

        transform = get_val_transforms(size=32)
        from PIL import Image
        img = Image.new("RGB", (32, 32), color=(128, 128, 128))
        tensor = transform(img)
        restored = denormalise(tensor)
        # Values should be roughly back to [0, 1] range
        assert restored.min() >= -0.1
        assert restored.max() <= 1.1


# ═══════════════════════════════════════════════════════════════
# URL Feature Extractor Tests
# ═══════════════════════════════════════════════════════════════

class TestURLFeatureExtractor:
    """Test URL feature extraction."""

    def test_feature_count(self):
        features = extract_url_features("https://www.google.com/search?q=test")
        assert len(features) == NUM_FEATURES

    def test_feature_vector_order(self):
        vec = extract_url_feature_vector("https://example.com")
        assert len(vec) == NUM_FEATURES
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)

    def test_safe_url_features(self):
        features = extract_url_features("https://www.example.com")
        assert features["has_https"] == 1.0
        assert features["has_ip_address"] == 0.0
        assert features["num_suspicious_words"] == 0.0

    def test_suspicious_url_features(self):
        features = extract_url_features(
            "http://192.168.1.1/login/verify-account.php?id=123"
        )
        assert features["has_https"] == 0.0
        assert features["has_ip_address"] == 1.0
        assert features["num_suspicious_words"] >= 2.0  # login, verify, account

    def test_phishing_url_high_entropy(self):
        # Random-looking domain = high entropy = suspicious
        features = extract_url_features(
            "http://x7k9m2p4q1r8.evil.com/paypal-login"
        )
        assert features["domain_entropy"] > 2.0

    def test_long_url_features(self):
        features = extract_url_features(
            "https://sub1.sub2.sub3.example.com/a/b/c/d?x=1&y=2&z=3"
        )
        assert features["num_subdomains"] >= 3.0
        assert features["num_params"] >= 3.0

    def test_empty_path_url(self):
        features = extract_url_features("https://example.com")
        assert features["path_length"] >= 0
        assert features["query_length"] == 0.0

    def test_feature_names_consistency(self):
        """Verify FEATURE_NAMES matches what extract_url_features returns."""
        features = extract_url_features("https://test.com")
        for name in FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"
