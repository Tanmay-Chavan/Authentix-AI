"""M3 Gate Tests — Model architecture and training loop verification."""

import pytest
import torch
import numpy as np

from authentix.models.image_classifier import ImageClassifier


class TestImageClassifier:
    """Verify EfficientNet-B4 classifier architecture."""

    @pytest.fixture
    def model(self):
        return ImageClassifier(
            num_classes=1,
            dropout=0.4,
            freeze_backbone=True,
            unfreeze_last_n_blocks=2,
            pretrained=False,  # Skip download in tests
        )

    def test_forward_shape(self, model):
        x = torch.randn(2, 3, 380, 380)
        logits = model(x)
        assert logits.shape == (2, 1)

    def test_predict_proba_range(self, model):
        x = torch.randn(4, 3, 380, 380)
        probs = model.predict_proba(x)
        assert probs.shape == (4,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_feature_maps_shape(self, model):
        x = torch.randn(2, 3, 380, 380)
        fmaps = model.get_feature_maps(x)
        # EfficientNet-B4 last block outputs 1792 channels
        assert fmaps.shape[1] == 1792
        assert fmaps.shape[0] == 2

    def test_frozen_params(self, model):
        """Verify backbone freezing works correctly."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_ratio = 1 - trainable / total
        # With 2 blocks unfrozen, ~60-70% should be frozen
        assert 0.5 < frozen_ratio < 0.8

    def test_classifier_head_trainable(self, model):
        """Verify the classifier head is always trainable."""
        for param in model.classifier.parameters():
            assert param.requires_grad is True

    def test_gradient_flow(self, model):
        """Verify gradients flow through unfrozen layers."""
        x = torch.randn(1, 3, 380, 380)
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        # Check classifier head has gradients
        for param in model.classifier.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_no_pretrained(self):
        """Model should work without pretrained weights."""
        model = ImageClassifier(pretrained=False)
        x = torch.randn(1, 3, 380, 380)
        logits = model(x)
        assert logits.shape == (1, 1)

    def test_multi_class(self):
        """Test multi-class mode (for potential future use)."""
        model = ImageClassifier(num_classes=3, pretrained=False)
        x = torch.randn(2, 3, 380, 380)
        logits = model(x)
        assert logits.shape == (2, 3)
        probs = model.predict_proba(x)
        # Multi-class: softmax, should sum to 1
        assert probs.shape == (2, 3)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)


class TestTrainerMetrics:
    """Verify metric computation logic."""

    def test_perfect_predictions(self):
        from scripts.train_image_classifier import Trainer
        metrics = Trainer._compute_metrics(
            labels=[0, 0, 1, 1],
            preds=[0, 0, 1, 1],
            probs=[0.1, 0.2, 0.9, 0.8],
        )
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["auc_roc"] == 1.0

    def test_worst_predictions(self):
        from scripts.train_image_classifier import Trainer
        metrics = Trainer._compute_metrics(
            labels=[0, 0, 1, 1],
            preds=[1, 1, 0, 0],
            probs=[0.9, 0.8, 0.1, 0.2],
        )
        assert metrics["accuracy"] == 0.0
        assert metrics["auc_roc"] == 0.0

    def test_confusion_matrix_shape(self):
        from scripts.train_image_classifier import Trainer
        metrics = Trainer._compute_metrics(
            labels=[0, 1, 0, 1],
            preds=[0, 1, 1, 0],
            probs=[0.3, 0.7, 0.6, 0.4],
        )
        cm = metrics["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2
