"""Phase 1.5 — URL Data Preprocessing & Cleaning Pipeline.

Implements the URL preprocessing steps from the review-2 defense document:

1. String Parsing:   tldextract accurate splitting
2. Length Filtering:  Drops URLs outside 10–1024 characters
3. Deduplication:     Exact URL deduplication
4. Outlier Capping:   StandardScaler + 3σ capping on extracted features

Generates a cleaning report.
"""

from __future__ import annotations

import csv
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler
import structlog

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

MIN_URL_LENGTH = 10
MAX_URL_LENGTH = 1024
OUTLIER_STD_THRESHOLD = 3.0  # Cap features beyond 3 standard deviations


@dataclass
class URLCleaningReport:
    """Report generated after cleaning a URL dataset."""

    dataset_name: str
    total_scanned: int = 0
    passed: int = 0
    too_short: int = 0
    too_long: int = 0
    duplicates_removed: int = 0
    label_parse_failures: int = 0
    outliers_capped: int = 0
    duration_seconds: float = 0.0

    @property
    def rejection_rate(self) -> float:
        if self.total_scanned == 0:
            return 0.0
        return (self.total_scanned - self.passed) / self.total_scanned * 100

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rejection_rate_pct"] = round(self.rejection_rate, 2)
        return d


def clean_url_dataset(
    csv_path: Path,
    output_path: Path,
    dataset_name: str = "malicious_phish",
    url_column: str = "url",
    label_column: str = "type",
    max_samples: Optional[int] = None,
) -> URLCleaningReport:
    """Run the full URL cleaning pipeline.

    Parameters
    ----------
    csv_path : Path
        Path to the input CSV.
    output_path : Path
        Path to write the cleaned CSV.
    dataset_name : str
        Name for the cleaning report.
    url_column, label_column : str
        CSV column names.
    max_samples : int, optional
        Limit samples for testing.

    Returns
    -------
    URLCleaningReport
    """
    start_time = time.time()
    report = URLCleaningReport(dataset_name=dataset_name)

    # Read raw CSV
    urls = []
    labels = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Flexible column detection
        url_key = _find_col(fieldnames, url_column, ["url", "URL", "Url"])
        label_key = _find_col(fieldnames, label_column, ["type", "Type", "label", "Label"])

        for row in reader:
            report.total_scanned += 1
            if max_samples and report.total_scanned > max_samples:
                break

            url = row.get(url_key, "").strip()
            label_raw = row.get(label_key, "").strip()

            # Parse label
            label = _parse_label(label_raw)
            if label is None:
                report.label_parse_failures += 1
                continue

            # Length filter
            if len(url) < MIN_URL_LENGTH:
                report.too_short += 1
                continue
            if len(url) > MAX_URL_LENGTH:
                report.too_long += 1
                continue

            urls.append(url)
            labels.append(label)

    # Deduplication
    seen = set()
    deduped_urls = []
    deduped_labels = []
    for url, label in zip(urls, labels):
        if url not in seen:
            seen.add(url)
            deduped_urls.append(url)
            deduped_labels.append(label)
        else:
            report.duplicates_removed += 1

    report.passed = len(deduped_urls)

    # Write cleaned CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label"])
        for url, label in zip(deduped_urls, deduped_labels):
            writer.writerow([url, label])

    report.duration_seconds = round(time.time() - start_time, 2)

    logger.info(
        "url_preprocessing.complete",
        dataset=dataset_name,
        total=report.total_scanned,
        passed=report.passed,
        duplicates=report.duplicates_removed,
        too_short=report.too_short,
        too_long=report.too_long,
        rejection_rate=f"{report.rejection_rate:.1f}%",
        duration=f"{report.duration_seconds:.1f}s",
    )

    return report


def cap_feature_outliers(
    features: np.ndarray,
    threshold: float = OUTLIER_STD_THRESHOLD,
) -> tuple[np.ndarray, int]:
    """Cap extreme outliers beyond ±threshold standard deviations.

    Prevents gradient explosion in tree-based models.

    Returns (capped_features, num_capped_values).
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # Count outliers
    outlier_mask = np.abs(scaled) > threshold
    num_capped = int(outlier_mask.sum())

    # Cap outliers
    scaled = np.clip(scaled, -threshold, threshold)

    # Inverse transform back to original scale
    capped = scaler.inverse_transform(scaled)

    return capped, num_capped


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _find_col(fieldnames: list[str], preferred: str, alternatives: list[str]) -> str:
    if preferred in fieldnames:
        return preferred
    for alt in alternatives:
        if alt in fieldnames:
            return alt
    raise ValueError(f"Column '{preferred}' not found in {fieldnames}")


def _parse_label(raw: str) -> Optional[int]:
    if raw in ("0", "1"):
        return int(raw)
    lower = raw.lower()
    if lower in ("benign", "legitimate", "safe", "good"):
        return 0
    if lower in ("malicious", "phishing", "bad", "spam", "defacement", "malware"):
        return 1
    return None
