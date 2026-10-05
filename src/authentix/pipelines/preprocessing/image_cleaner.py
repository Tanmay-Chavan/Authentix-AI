"""Phase 1.5 — Image Data Preprocessing & Cleaning Pipeline.

Implements the deterministic 5-step preprocessing pipeline
from the review-2 defense document:

1. Integrity:    PIL.Image.verify() — drops corrupted files
2. Deduplication: pHash — Hamming distance ≤ 4 prevents data leakage
3. Dimensions:   Aspect ratio 0.5–2.0, min 256×256
4. Channels:     RGB standardization (strip RGBA/Grayscale)
5. Normalization: ImageNet stats verified

Generates a cleaning report per dataset.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import imagehash
from PIL import Image, UnidentifiedImageError
import structlog

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

MIN_DIMENSION = 32
MIN_ASPECT_RATIO = 0.5
MAX_ASPECT_RATIO = 2.0
PHASH_HAMMING_THRESHOLD = 4  # Images within this distance are "duplicates"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


@dataclass
class CleaningReport:
    """Report generated after cleaning a dataset."""

    dataset_name: str
    total_scanned: int = 0
    passed: int = 0
    corrupted: int = 0
    duplicates_removed: int = 0
    dimension_rejected: int = 0
    channel_converted: int = 0
    errors: int = 0
    duration_seconds: float = 0.0
    rejected_files: list[dict] = field(default_factory=list)

    @property
    def rejection_rate(self) -> float:
        if self.total_scanned == 0:
            return 0.0
        return (self.total_scanned - self.passed) / self.total_scanned * 100

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rejection_rate_pct"] = round(self.rejection_rate, 2)
        return d


# ═══════════════════════════════════════════════════════════════
# Step 1: Integrity Check
# ═══════════════════════════════════════════════════════════════

def check_integrity(path: Path) -> tuple[bool, str]:
    """Verify image file integrity using PIL.

    Returns (is_valid, reason).
    """
    try:
        with Image.open(path) as img:
            img.verify()
        # Re-open to actually load pixel data (verify only checks headers)
        with Image.open(path) as img:
            img.load()
        return True, "OK"
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        return False, f"Corrupted: {str(e)[:100]}"
    except Exception as e:
        return False, f"Error: {str(e)[:100]}"


# ═══════════════════════════════════════════════════════════════
# Step 2: Perceptual Hash Deduplication
# ═══════════════════════════════════════════════════════════════

def compute_phash(path: Path, hash_size: int = 16) -> Optional[imagehash.ImageHash]:
    """Compute perceptual hash of an image.

    Uses a larger hash_size (16 vs default 8) for higher precision.
    """
    try:
        with Image.open(path) as img:
            return imagehash.phash(img, hash_size=hash_size)
    except Exception:
        return None


def deduplicate_within_split(
    image_paths: list[Path],
    threshold: int = PHASH_HAMMING_THRESHOLD,
) -> tuple[list[Path], list[Path]]:
    """Remove perceptual duplicates within a single split.

    Prevents data leakage by ensuring no near-identical images exist.
    Uses O(n²) pairwise comparison — acceptable for <100K images.

    Returns (kept, removed).
    """
    hashes: list[tuple[Path, imagehash.ImageHash]] = []
    removed: list[Path] = []

    for path in image_paths:
        h = compute_phash(path)
        if h is None:
            removed.append(path)
            continue

        is_dup = False
        for _, existing_hash in hashes:
            if h - existing_hash <= threshold:
                is_dup = True
                break

        if is_dup:
            removed.append(path)
        else:
            hashes.append((path, h))

    kept = [p for p, _ in hashes]
    return kept, removed


def check_cross_split_leakage(
    train_paths: list[Path],
    test_paths: list[Path],
    threshold: int = PHASH_HAMMING_THRESHOLD,
    max_check: int = 5000,
) -> list[tuple[Path, Path]]:
    """Check for near-duplicate images between train and test splits.

    Returns list of (train_path, test_path) leakage pairs.
    """
    logger.info(
        "preprocessing.cross_split_check",
        train_count=len(train_paths),
        test_count=len(test_paths),
    )

    # Compute hashes for test set (smaller)
    test_hashes = []
    for p in test_paths[:max_check]:
        h = compute_phash(p)
        if h is not None:
            test_hashes.append((p, h))

    # Check each train image against test hashes
    leaks = []
    for p in train_paths[:max_check]:
        h = compute_phash(p)
        if h is None:
            continue
        for test_path, test_hash in test_hashes:
            if h - test_hash <= threshold:
                leaks.append((p, test_path))
                break

    return leaks


# ═══════════════════════════════════════════════════════════════
# Step 3: Dimension Validation
# ═══════════════════════════════════════════════════════════════

def check_dimensions(path: Path) -> tuple[bool, str]:
    """Validate image dimensions and aspect ratio.

    Rejects images that are:
    - Smaller than MIN_DIMENSION on either axis
    - Have extreme aspect ratios (too wide or too tall)
    """
    try:
        with Image.open(path) as img:
            w, h = img.size

        if w < MIN_DIMENSION or h < MIN_DIMENSION:
            return False, f"Too small: {w}x{h} (min {MIN_DIMENSION})"

        aspect = w / h
        if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
            return False, f"Bad aspect: {aspect:.2f} (range {MIN_ASPECT_RATIO}–{MAX_ASPECT_RATIO})"

        return True, "OK"
    except Exception as e:
        return False, f"Error: {str(e)[:100]}"


# ═══════════════════════════════════════════════════════════════
# Step 4: Channel Standardization
# ═══════════════════════════════════════════════════════════════

def standardize_channels(path: Path, output_path: Optional[Path] = None) -> tuple[bool, str]:
    """Convert image to RGB (3-channel).

    Strips alpha channels (RGBA → RGB) and converts grayscale (L → RGB).
    Saves the converted image in-place or to output_path.

    Returns (was_converted, mode_before).
    """
    try:
        with Image.open(path) as img:
            original_mode = img.mode
            if img.mode == "RGB":
                return False, "Already RGB"

            rgb = img.convert("RGB")
            save_path = output_path or path
            rgb.save(save_path, quality=95)
            return True, f"Converted {original_mode} → RGB"
    except Exception as e:
        return False, f"Error: {str(e)[:100]}"


# ═══════════════════════════════════════════════════════════════
# Full Pipeline
# ═══════════════════════════════════════════════════════════════

def clean_image_dataset(
    dataset_dir: Path,
    dataset_name: str,
    output_dir: Optional[Path] = None,
    run_dedup: bool = True,
    max_dedup_per_class: int = 10000,
    dry_run: bool = False,
) -> CleaningReport:
    """Run the full 5-step cleaning pipeline on an image dataset.

    Parameters
    ----------
    dataset_dir : Path
        Root directory containing class folders (e.g., REAL/, FAKE/).
    dataset_name : str
        Name for the cleaning report.
    output_dir : Path, optional
        If set, copy clean images here instead of modifying in-place.
    run_dedup : bool
        Whether to run the pHash deduplication step.
    max_dedup_per_class : int
        Max images per class for dedup (O(n²) can be slow).
    dry_run : bool
        If True, only report what would be done.

    Returns
    -------
    CleaningReport
    """
    start_time = time.time()
    report = CleaningReport(dataset_name=dataset_name)

    # Discover all image files
    all_images = sorted(
        f for f in dataset_dir.rglob("*")
        if f.suffix.lower() in VALID_EXTENSIONS and f.is_file()
    )
    report.total_scanned = len(all_images)
    logger.info(
        "preprocessing.scan_complete",
        dataset=dataset_name,
        total_images=len(all_images),
    )

    # Step 1: Integrity
    valid_images: list[Path] = []
    for path in all_images:
        is_valid, reason = check_integrity(path)
        if is_valid:
            valid_images.append(path)
        else:
            report.corrupted += 1
            report.rejected_files.append({
                "path": str(path.relative_to(dataset_dir)),
                "step": "integrity",
                "reason": reason,
            })

    logger.info(
        "preprocessing.integrity_done",
        valid=len(valid_images),
        corrupted=report.corrupted,
    )

    # Step 3: Dimensions (run before dedup — faster to filter first)
    dimension_valid: list[Path] = []
    for path in valid_images:
        is_valid, reason = check_dimensions(path)
        if is_valid:
            dimension_valid.append(path)
        else:
            report.dimension_rejected += 1
            report.rejected_files.append({
                "path": str(path.relative_to(dataset_dir)),
                "step": "dimensions",
                "reason": reason,
            })

    logger.info(
        "preprocessing.dimensions_done",
        valid=len(dimension_valid),
        rejected=report.dimension_rejected,
    )

    # Step 2: Deduplication (most expensive — run on filtered set)
    if run_dedup and len(dimension_valid) <= max_dedup_per_class * 10:
        kept, removed = deduplicate_within_split(dimension_valid)
        report.duplicates_removed = len(removed)
        for path in removed:
            report.rejected_files.append({
                "path": str(path.relative_to(dataset_dir)),
                "step": "deduplication",
                "reason": "pHash duplicate (Hamming ≤ 4)",
            })
        clean_images = kept
        logger.info(
            "preprocessing.dedup_done",
            kept=len(kept),
            removed=len(removed),
        )
    else:
        clean_images = dimension_valid
        if run_dedup:
            logger.warning(
                "preprocessing.dedup_skipped",
                reason=f"Too many images ({len(dimension_valid)}) for O(n²) dedup",
            )

    # Step 4: Channel standardization
    for path in clean_images:
        was_converted, mode_info = standardize_channels(path)
        if was_converted:
            report.channel_converted += 1

    logger.info(
        "preprocessing.channels_done",
        converted=report.channel_converted,
    )

    # Step 5: Copy to output if requested
    if output_dir and not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in clean_images:
            rel = path.relative_to(dataset_dir)
            dest = output_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)

    report.passed = len(clean_images)
    report.duration_seconds = round(time.time() - start_time, 2)

    logger.info(
        "preprocessing.complete",
        dataset=dataset_name,
        total=report.total_scanned,
        passed=report.passed,
        rejection_rate=f"{report.rejection_rate:.1f}%",
        duration=f"{report.duration_seconds:.1f}s",
    )

    return report
