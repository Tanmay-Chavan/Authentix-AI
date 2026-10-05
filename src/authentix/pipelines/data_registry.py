"""Unified data registry — single entry point for all dataset loading.

Provides a clean API for downstream code to load any dataset
without knowing the details of directory layout or file format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import structlog
from torch.utils.data import DataLoader

from authentix.core.config import settings
from authentix.pipelines.image.datasets import CIFAKEDataset, TinyGenImageDataset
from authentix.pipelines.image.transforms import (
    get_train_transforms,
    get_val_transforms,
)
from authentix.pipelines.url.datasets import URLDataset

logger = structlog.get_logger(__name__)


class DataRegistry:
    """Central data registry — resolves dataset names to loaded datasets.

    Usage:
        registry = DataRegistry(data_dir="d:/Files/DMCE/Authentix/data")
        train_loader = registry.get_image_dataloader("cifake", split="train")
        url_X, url_y = registry.get_url_features("malicious_phish")
    """

    def __init__(self, data_dir: Optional[str | Path] = None):
        self.data_dir = Path(data_dir or settings.data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    # ── Image Datasets ──

    def get_cifake(
        self,
        split: Literal["train", "test"] = "train",
        transform=None,
        max_samples_per_class: Optional[int] = None,
    ) -> CIFAKEDataset:
        """Load CIFAKE dataset."""
        if transform is None:
            transform = (
                get_train_transforms() if split == "train"
                else get_val_transforms()
            )
        return CIFAKEDataset(
            root=self.data_dir / "CIFAKE",
            split=split,
            transform=transform,
            max_samples_per_class=max_samples_per_class,
        )

    def get_genimage(
        self,
        split: Literal["train", "val"] = "train",
        generators: Optional[list[str]] = None,
        transform=None,
        max_samples_per_class: Optional[int] = None,
    ) -> TinyGenImageDataset:
        """Load Tiny GenImage dataset."""
        if transform is None:
            transform = (
                get_train_transforms() if split == "train"
                else get_val_transforms()
            )
        return TinyGenImageDataset(
            root=self.data_dir / "tiny_genimage",
            split=split,
            generators=generators,
            transform=transform,
            max_samples_per_class=max_samples_per_class,
        )

    def get_image_dataloader(
        self,
        dataset_name: Literal["cifake", "genimage"],
        split: str = "train",
        batch_size: int = 4,
        num_workers: int = 0,
        shuffle: Optional[bool] = None,
        **kwargs: Any,
    ) -> DataLoader:
        """Get a ready-to-use DataLoader for image training/eval.

        Parameters
        ----------
        dataset_name : str
            "cifake" or "genimage".
        split : str
            "train", "test", or "val" depending on dataset.
        batch_size : int
            Batch size (keep small for 4GB VRAM).
        num_workers : int
            Number of data loading workers. 0 = main process.
        shuffle : bool, optional
            Whether to shuffle. Defaults to True for train, False otherwise.
        **kwargs
            Additional kwargs passed to the dataset constructor.
        """
        if shuffle is None:
            shuffle = (split == "train")

        if dataset_name == "cifake":
            dataset = self.get_cifake(split=split, **kwargs)
        elif dataset_name == "genimage":
            dataset = self.get_genimage(split=split, **kwargs)
        else:
            raise ValueError(f"Unknown image dataset: {dataset_name}")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )

    # ── URL Datasets ──

    def get_url_dataset(
        self,
        csv_name: str = "malicious_phish.csv",
        max_samples: Optional[int] = None,
    ) -> URLDataset:
        """Load a URL dataset from the data/urls/ directory."""
        return URLDataset(
            csv_path=self.data_dir / "urls" / csv_name,
            url_column="url",
            label_column="type",
            max_samples=max_samples,
        )

    def get_url_features(
        self,
        csv_name: str = "malicious_phish.csv",
        max_samples: Optional[int] = None,
    ):
        """Get (X, y) feature arrays ready for ML training.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            X: (n_samples, 33), y: (n_samples,)
        """
        dataset = self.get_url_dataset(csv_name, max_samples)
        return dataset.get_features_and_labels()

    # ── Info ──

    def list_available(self) -> dict[str, bool]:
        """Check which datasets are available on disk."""
        return {
            "cifake": (self.data_dir / "CIFAKE").exists(),
            "tiny_genimage": (self.data_dir / "tiny_genimage").exists(),
            "malicious_urls": (self.data_dir / "urls" / "malicious_phish.csv").exists(),
            "tranco_top1m": (self.data_dir / "urls" / "top-1m.csv").exists(),
            "faceforensics": (self.data_dir / "faceforensics").exists(),
        }
