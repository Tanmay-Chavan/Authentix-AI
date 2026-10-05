"""Standard image transforms for training and inference.

Centralises all augmentation and preprocessing logic so that
every pipeline uses identical transforms — critical for reproducibility.
"""

from __future__ import annotations

from torchvision import transforms


# ── EfficientNet-B4 expects 380×380 input ──
EFFICIENTNET_B4_SIZE = 380

# ── ImageNet normalisation constants ──
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(
    size: int = EFFICIENTNET_B4_SIZE,
    mean: list[float] = IMAGENET_MEAN,
    std: list[float] = IMAGENET_STD,
) -> transforms.Compose:
    """Training transforms with data augmentation.

    Augmentations are deliberately conservative — heavy augmentation
    can destroy subtle forensic artefacts that distinguish real from
    AI-generated images.
    """
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.02,
        ),
        # Mild JPEG compression simulation — important for robustness
        # to social media re-compression
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
            p=0.1,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_val_transforms(
    size: int = EFFICIENTNET_B4_SIZE,
    mean: list[float] = IMAGENET_MEAN,
    std: list[float] = IMAGENET_STD,
) -> transforms.Compose:
    """Validation / inference transforms — NO augmentation."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_inference_transforms(
    size: int = EFFICIENTNET_B4_SIZE,
    mean: list[float] = IMAGENET_MEAN,
    std: list[float] = IMAGENET_STD,
) -> transforms.Compose:
    """Inference transforms — identical to val but named explicitly."""
    return get_val_transforms(size=size, mean=mean, std=std)


def denormalise(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Reverse ImageNet normalisation for visualisation.

    Useful for Grad-CAM overlay rendering.
    """
    import torch

    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)

    if tensor.device != mean_t.device:
        mean_t = mean_t.to(tensor.device)
        std_t = std_t.to(tensor.device)

    return tensor * std_t + mean_t
