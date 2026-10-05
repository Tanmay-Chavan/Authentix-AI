"""URL dataset loader for malicious URL detection.

Loads CSV datasets of URLs with labels for XGBoost training.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import structlog

from authentix.pipelines.url.feature_extractor import (
    extract_url_feature_vector,
    NUM_FEATURES,
    FEATURE_NAMES,
)

logger = structlog.get_logger(__name__)


class URLDataset:
    """URL dataset — loads URLs from CSV and extracts feature vectors.

    Expected CSV format:
        url,type,label
        https://www.google.com,benign,0
        http://evil-phishing.xyz/login,phishing,1

    Or simpler format:
        url,label
        https://www.google.com,0
        http://evil-phishing.xyz/login,1
    """

    def __init__(
        self,
        csv_path: str | Path,
        url_column: str = "url",
        label_column: str = "label",
        max_samples: Optional[int] = None,
        precompute_features: bool = True,
    ):
        """
        Parameters
        ----------
        csv_path : str | Path
            Path to the CSV file containing URLs and labels.
        url_column : str
            Name of the column containing URLs.
        label_column : str
            Name of the column containing labels (0 = benign, 1 = malicious).
        max_samples : int, optional
            Limit the number of samples to load.
        precompute_features : bool
            If True, extract features for all URLs on init.
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"URL dataset not found: {self.csv_path}")

        self.urls: list[str] = []
        self.labels: list[int] = []
        self._load_csv(url_column, label_column, max_samples)

        self.features: Optional[np.ndarray] = None
        if precompute_features:
            self._extract_all_features()

        logger.info(
            "dataset.url.loaded",
            path=str(self.csv_path),
            total_samples=len(self.urls),
            benign_count=sum(1 for l in self.labels if l == 0),
            malicious_count=sum(1 for l in self.labels if l == 1),
        )

    def _load_csv(
        self,
        url_col: str,
        label_col: str,
        max_samples: Optional[int],
    ) -> None:
        """Load URLs and labels from CSV."""
        with open(self.csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)

            # Auto-detect column names
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no headers: {self.csv_path}")

            # Flexible column detection
            url_key = self._find_column(reader.fieldnames, url_col, ["url", "URL", "Url", "urls"])
            label_key = self._find_column(reader.fieldnames, label_col, ["label", "Label", "type", "Type", "class"])

            count = 0
            for row in reader:
                if max_samples is not None and count >= max_samples:
                    break

                url = row.get(url_key, "").strip()
                label_raw = row.get(label_key, "").strip()

                if not url:
                    continue

                # Parse label — handle both numeric and string labels
                label = self._parse_label(label_raw)
                if label is None:
                    continue

                self.urls.append(url)
                self.labels.append(label)
                count += 1

    def _find_column(
        self,
        fieldnames: list[str],
        preferred: str,
        alternatives: list[str],
    ) -> str:
        """Find a column name, trying preferred then alternatives."""
        if preferred in fieldnames:
            return preferred
        for alt in alternatives:
            if alt in fieldnames:
                return alt
        raise ValueError(
            f"Could not find column '{preferred}' or alternatives "
            f"{alternatives} in CSV columns: {fieldnames}"
        )

    @staticmethod
    def _parse_label(raw: str) -> Optional[int]:
        """Parse a label value, handling numeric and string formats."""
        if raw in ("0", "1"):
            return int(raw)
        lower = raw.lower()
        if lower in ("benign", "legitimate", "safe", "good"):
            return 0
        if lower in ("malicious", "phishing", "bad", "spam", "defacement",
                      "malware"):
            return 1
        return None

    def _extract_all_features(self) -> None:
        """Extract features for all URLs (can be slow for large datasets)."""
        logger.info("dataset.url.extracting_features", count=len(self.urls))
        features = []
        failed = 0
        for url in self.urls:
            try:
                vec = extract_url_feature_vector(url)
                features.append(vec)
            except Exception:
                features.append([0.0] * NUM_FEATURES)
                failed += 1

        self.features = np.array(features, dtype=np.float32)
        if failed > 0:
            logger.warning(
                "dataset.url.feature_extraction_failures",
                failed=failed,
                total=len(self.urls),
            )

    def get_features_and_labels(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) arrays ready for scikit-learn / XGBoost.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            X: shape (n_samples, NUM_FEATURES), dtype float32
            y: shape (n_samples,), dtype int32
        """
        if self.features is None:
            self._extract_all_features()
        return self.features, np.array(self.labels, dtype=np.int32)

    @property
    def class_weights(self) -> dict[int, float]:
        """Compute class weights for imbalanced datasets."""
        counts = {0: 0, 1: 0}
        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1
        total = sum(counts.values())
        n_classes = len(counts)
        return {
            cls: total / (n_classes * count) if count > 0 else 1.0
            for cls, count in counts.items()
        }

    def __len__(self) -> int:
        return len(self.urls)
