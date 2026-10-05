"""Image dataset loaders for CIFAKE and Tiny GenImage.

Provides PyTorch Dataset classes with configurable transforms,
train/val/test splits, and cross-generator evaluation support.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import structlog

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# CIFAKE Dataset
# ═══════════════════════════════════════════════════════════════

class CIFAKEDataset(Dataset):
    """CIFAKE dataset — binary classification (REAL vs FAKE).

    Expected directory structure:
        data/CIFAKE/
        ├── train/
        │   ├── REAL/
        │   └── FAKE/
        └── test/
            ├── REAL/
            └── FAKE/

    Labels: 0 = REAL, 1 = FAKE
    """

    LABEL_MAP = {"REAL": 0, "FAKE": 1}

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "test"] = "train",
        transform: Optional[Callable] = None,
        max_samples_per_class: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        root : str | Path
            Path to the CIFAKE root directory (containing train/ and test/).
        split : str
            "train" or "test".
        transform : callable, optional
            Torchvision transform to apply to each image.
        max_samples_per_class : int, optional
            If set, limit the number of samples per class (for debugging / quick runs).
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.split_dir = self.root / split

        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"CIFAKE split directory not found: {self.split_dir}"
            )

        self.samples: list[tuple[Path, int]] = []
        self._load_samples(max_samples_per_class)

        logger.info(
            "dataset.cifake.loaded",
            split=split,
            total_samples=len(self.samples),
            real_count=sum(1 for _, l in self.samples if l == 0),
            fake_count=sum(1 for _, l in self.samples if l == 1),
        )

    def _load_samples(self, max_per_class: Optional[int]) -> None:
        """Scan directory and collect (path, label) pairs."""
        for class_name, label in self.LABEL_MAP.items():
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                logger.warning(
                    "dataset.cifake.class_missing",
                    class_name=class_name,
                    path=str(class_dir),
                )
                continue

            files = sorted(
                f for f in class_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            )
            if max_per_class is not None:
                files = files[:max_per_class]

            self.samples.extend((f, label) for f in files)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "path": str(path),
        }

    @property
    def class_weights(self) -> list[float]:
        """Compute inverse frequency class weights for balanced training."""
        counts = [0, 0]
        for _, label in self.samples:
            counts[label] += 1
        total = sum(counts)
        return [total / (2 * c) if c > 0 else 1.0 for c in counts]


# ═══════════════════════════════════════════════════════════════
# Tiny GenImage Dataset
# ═══════════════════════════════════════════════════════════════

# Known generator directory names in Tiny GenImage
GENIMAGE_GENERATORS = [
    "imagenet_ai_0419_biggan",
    "imagenet_ai_0419_vqdm",
    "imagenet_ai_0424_sdv5",
    "imagenet_ai_0424_wukong",
    "imagenet_ai_0508_adm",
    "imagenet_glide",
    "imagenet_midjourney",
]


class TinyGenImageDataset(Dataset):
    """Tiny GenImage dataset — multi-generator AI image detection.

    Expected directory structure:
        data/tiny_genimage/
        ├── imagenet_ai_0419_biggan/
        │   ├── train/
        │   │   ├── ai/
        │   │   └── nature/
        │   └── val/
        │       ├── ai/
        │       └── nature/
        ├── imagenet_ai_0424_sdv5/
        │   └── ...
        └── ...

    Labels: 0 = nature (real), 1 = ai (generated)
    """

    LABEL_MAP = {"nature": 0, "ai": 1}

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val"] = "train",
        generators: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        max_samples_per_class: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        root : str | Path
            Path to the tiny_genimage root directory.
        split : str
            "train" or "val".
        generators : list[str], optional
            Subset of generators to load. Defaults to all available.
            Use this for cross-generator evaluation (train on some, test on others).
        transform : callable, optional
            Torchvision transform to apply to each image.
        max_samples_per_class : int, optional
            Limit samples per (generator, class) pair.
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.generators = generators or self._discover_generators()

        if not self.root.exists():
            raise FileNotFoundError(
                f"Tiny GenImage root not found: {self.root}"
            )

        self.samples: list[tuple[Path, int, str]] = []  # (path, label, generator)
        self._load_samples(max_samples_per_class)

        logger.info(
            "dataset.genimage.loaded",
            split=split,
            generators=self.generators,
            total_samples=len(self.samples),
            real_count=sum(1 for _, l, _ in self.samples if l == 0),
            fake_count=sum(1 for _, l, _ in self.samples if l == 1),
        )

    def _discover_generators(self) -> list[str]:
        """Auto-discover available generator directories."""
        found = []
        for gen in GENIMAGE_GENERATORS:
            if (self.root / gen).exists():
                found.append(gen)
        if not found:
            raise FileNotFoundError(
                f"No generator directories found in {self.root}. "
                f"Expected: {GENIMAGE_GENERATORS}"
            )
        return found

    def _load_samples(self, max_per_class: Optional[int]) -> None:
        """Scan all generator directories and collect samples."""
        for gen_name in self.generators:
            for class_name, label in self.LABEL_MAP.items():
                class_dir = self.root / gen_name / self.split / class_name
                if not class_dir.exists():
                    logger.warning(
                        "dataset.genimage.class_missing",
                        generator=gen_name,
                        class_name=class_name,
                        path=str(class_dir),
                    )
                    continue

                files = sorted(
                    f for f in class_dir.iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                )
                if max_per_class is not None:
                    files = files[:max_per_class]

                self.samples.extend((f, label, gen_name) for f in files)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path, label, generator = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "generator": generator,
            "path": str(path),
        }

    @property
    def class_weights(self) -> list[float]:
        """Compute inverse frequency class weights for balanced training."""
        counts = [0, 0]
        for _, label, _ in self.samples:
            counts[label] += 1
        total = sum(counts)
        return [total / (2 * c) if c > 0 else 1.0 for c in counts]

    def get_generator_distribution(self) -> dict[str, dict[str, int]]:
        """Return sample counts per generator per class."""
        dist: dict[str, dict[str, int]] = {}
        for _, label, gen in self.samples:
            if gen not in dist:
                dist[gen] = {"real": 0, "ai": 0}
            key = "real" if label == 0 else "ai"
            dist[gen][key] += 1
        return dist
