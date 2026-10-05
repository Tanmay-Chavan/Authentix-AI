"""EfficientNet-B4 based image authenticity classifier.

Fine-tunes a pre-trained EfficientNet-B4 for binary classification
(real vs AI-generated / manipulated). Uses transfer learning so
training converges in 3-5 epochs on modest hardware.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
import structlog

logger = structlog.get_logger(__name__)


class ImageClassifier(nn.Module):
    """EfficientNet-B4 fine-tuned for real vs fake image detection.

    Architecture:
        EfficientNet-B4 backbone (frozen early layers)
        → Adaptive Average Pool
        → Dropout(0.4)
        → Linear(1792, 512) + ReLU + Dropout(0.3)
        → Linear(512, 1)  [logit output]

    The backbone's early layers (feature extractors for edges, textures)
    are frozen — we only train the classifier head and the last few
    convolutional blocks. This is critical because:
    1. We have limited data compared to ImageNet.
    2. Training is fast (hours, not days).
    3. The model retains its general visual understanding.
    """

    def __init__(
        self,
        num_classes: int = 1,
        dropout: float = 0.4,
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 2,
        pretrained: bool = True,
    ):
        """
        Parameters
        ----------
        num_classes : int
            1 for binary (sigmoid output), >1 for multi-class.
        dropout : float
            Dropout rate before the classifier head.
        freeze_backbone : bool
            Whether to freeze the backbone initially.
        unfreeze_last_n_blocks : int
            Number of EfficientNet blocks to unfreeze from the end.
            Set to 2 for fine-tuning with limited data.
        pretrained : bool
            Load ImageNet pre-trained weights.
        """
        super().__init__()

        # Load pre-trained EfficientNet-B4
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b4(weights=weights)

        # Get the feature dimension from the existing classifier
        in_features = self.backbone.classifier[1].in_features  # 1792

        # Replace the classifier head
        self.backbone.classifier = nn.Identity()

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.75),  # Lighter dropout in middle
            nn.Linear(512, num_classes),
        )

        # Freeze/unfreeze strategy
        if freeze_backbone:
            self._freeze_backbone(unfreeze_last_n_blocks)

        # Log architecture summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        logger.info(
            "model.image_classifier.init",
            total_params=f"{total_params:,}",
            trainable_params=f"{trainable_params:,}",
            frozen_pct=f"{(1 - trainable_params / total_params) * 100:.1f}%",
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )

    def _freeze_backbone(self, unfreeze_last_n: int) -> None:
        """Freeze all backbone parameters except the last N blocks."""
        # Freeze everything first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last N blocks of the features
        features = list(self.backbone.features.children())
        for block in features[-unfreeze_last_n:]:
            for param in block.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input images, shape (B, 3, 380, 380).

        Returns
        -------
        torch.Tensor
            Logits, shape (B, num_classes). Apply sigmoid for probabilities.
        """
        features = self.backbone(x)  # (B, 1792)
        logits = self.classifier(features)  # (B, num_classes)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability of being fake/AI-generated.

        Returns
        -------
        torch.Tensor
            Probabilities in [0, 1], shape (B,).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            if logits.shape[-1] == 1:
                return torch.sigmoid(logits).squeeze(-1)
            return torch.softmax(logits, dim=-1)

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature maps from the last conv layer (for Grad-CAM).

        Returns
        -------
        torch.Tensor
            Feature maps from the last convolutional block.
        """
        # Hook into the last feature block
        features = None

        def hook_fn(module, input, output):
            nonlocal features
            features = output

        # Register hook on the last features block
        handle = self.backbone.features[-1].register_forward_hook(hook_fn)
        _ = self.backbone(x)
        handle.remove()

        return features

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        device: str = "cpu",
        **kwargs,
    ) -> "ImageClassifier":
        """Load a model from a saved checkpoint.

        Parameters
        ----------
        path : str
            Path to the .pt checkpoint file.
        device : str
            Device to load the model on.
        """
        model = cls(pretrained=False, **kwargs)
        checkpoint = torch.load(path, map_location=device, weights_only=True)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        logger.info("model.image_classifier.loaded", path=path, device=device)
        return model
