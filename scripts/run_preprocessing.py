"""Master Preprocessing Script — Phase 1.5

Runs the complete data cleaning pipeline across all modalities:
1. Image datasets (CIFAKE, Tiny GenImage)
2. Video dataset (FaceForensics++)
3. URL dataset (Malicious Phish)

Generates per-modality cleaning reports in data/reports/.

Usage:
    python scripts/run_preprocessing.py
    python scripts/run_preprocessing.py --skip-video  # Skip slow video processing
    python scripts/run_preprocessing.py --images-only  # Only clean images
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from authentix.pipelines.preprocessing.image_cleaner import (
    clean_image_dataset,
    check_cross_split_leakage,
)
from authentix.pipelines.preprocessing.video_cleaner import (
    preprocess_video_dataset,
)
from authentix.pipelines.preprocessing.url_cleaner import (
    clean_url_dataset,
)

import structlog
logger = structlog.get_logger(__name__)


DATA_DIR = Path("d:/Files/DMCE/Authentix/data")
REPORT_DIR = DATA_DIR / "reports"


def run_image_preprocessing():
    """Clean all image datasets."""
    print("\n" + "=" * 60)
    print("🖼️  IMAGE PREPROCESSING")
    print("=" * 60)

    reports = []

    # ── CIFAKE ──
    cifake_dir = DATA_DIR / "CIFAKE"
    if cifake_dir.exists():
        for split in ["train", "test"]:
            split_dir = cifake_dir / split
            if split_dir.exists():
                print(f"\n  Cleaning CIFAKE/{split}...")
                report = clean_image_dataset(
                    split_dir,
                    dataset_name=f"cifake_{split}",
                    run_dedup=True,
                    max_dedup_per_class=5000,
                )
                reports.append(report)
                _print_image_report(report)

        # Cross-split leakage check
        train_real = list((cifake_dir / "train" / "REAL").glob("*.*"))
        test_real = list((cifake_dir / "test" / "REAL").glob("*.*"))
        if train_real and test_real:
            print("\n  Checking cross-split leakage (CIFAKE)...")
            leaks = check_cross_split_leakage(
                train_real[:2000], test_real[:2000]
            )
            print(f"  → Leakage pairs found: {len(leaks)}")
            if leaks:
                print(f"    ⚠️ WARNING: {len(leaks)} near-duplicate images across train/test!")

    # ── Tiny GenImage ──
    genimage_dir = DATA_DIR / "tiny_genimage"
    if genimage_dir.exists():
        for gen_dir in sorted(genimage_dir.iterdir()):
            if gen_dir.is_dir():
                for split in ["train", "val"]:
                    split_dir = gen_dir / split
                    if split_dir.exists():
                        print(f"\n  Cleaning GenImage/{gen_dir.name}/{split}...")
                        report = clean_image_dataset(
                            split_dir,
                            dataset_name=f"genimage_{gen_dir.name}_{split}",
                            run_dedup=True,
                            max_dedup_per_class=3000,
                        )
                        reports.append(report)
                        _print_image_report(report)

    return reports


def run_video_preprocessing():
    """Clean all video datasets."""
    print("\n" + "=" * 60)
    print("🎬  VIDEO PREPROCESSING")
    print("=" * 60)

    reports = []
    ff_dir = DATA_DIR / "faceforensics"

    # ── FF++ Originals ──
    orig_dir = ff_dir / "original_sequences" / "youtube" / "c23" / "videos"
    if orig_dir.exists():
        print(f"\n  Processing FF++ Originals ({len(list(orig_dir.glob('*.mp4')))} videos)...")
        report = preprocess_video_dataset(
            orig_dir,
            output_dir=DATA_DIR / "processed" / "faceforensics" / "original",
            dataset_name="ff++_original",
            with_face_crop=True,
        )
        reports.append(report)
        _print_video_report(report)

    # ── FF++ Deepfakes ──
    fake_dir = ff_dir / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
    if fake_dir.exists():
        print(f"\n  Processing FF++ Deepfakes ({len(list(fake_dir.glob('*.mp4')))} videos)...")
        report = preprocess_video_dataset(
            fake_dir,
            output_dir=DATA_DIR / "processed" / "faceforensics" / "deepfakes",
            dataset_name="ff++_deepfakes",
            with_face_crop=True,
        )
        reports.append(report)
        _print_video_report(report)

    return reports


def run_url_preprocessing():
    """Clean URL dataset."""
    print("\n" + "=" * 60)
    print("🌐  URL PREPROCESSING")
    print("=" * 60)

    reports = []
    url_csv = DATA_DIR / "urls" / "malicious_phish.csv"

    if url_csv.exists():
        print(f"\n  Cleaning {url_csv.name}...")
        report = clean_url_dataset(
            url_csv,
            output_path=DATA_DIR / "processed" / "urls" / "malicious_phish_clean.csv",
            dataset_name="malicious_phish",
        )
        reports.append(report)
        _print_url_report(report)

    return reports


def _print_image_report(report):
    print(f"    Total: {report.total_scanned:,}")
    print(f"    Passed: {report.passed:,} ({100 - report.rejection_rate:.1f}%)")
    print(f"    Corrupted: {report.corrupted}")
    print(f"    Dim rejected: {report.dimension_rejected}")
    print(f"    Duplicates: {report.duplicates_removed}")
    print(f"    Channels converted: {report.channel_converted}")
    print(f"    Time: {report.duration_seconds:.1f}s")


def _print_video_report(report):
    print(f"    Videos: {report.valid_videos}/{report.total_videos} valid")
    print(f"    Corrupted: {report.corrupted_videos}")
    print(f"    Frames extracted: {report.total_frames_extracted:,}")
    print(f"    Blurry dropped: {report.blurry_frames_dropped}")
    print(f"    Time: {report.duration_seconds:.1f}s")


def _print_url_report(report):
    print(f"    Total: {report.total_scanned:,}")
    print(f"    Passed: {report.passed:,} ({100 - report.rejection_rate:.1f}%)")
    print(f"    Too short: {report.too_short}")
    print(f"    Too long: {report.too_long}")
    print(f"    Duplicates: {report.duplicates_removed}")
    print(f"    Label failures: {report.label_parse_failures}")
    print(f"    Time: {report.duration_seconds:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Phase 1.5 Data Preprocessing")
    parser.add_argument("--skip-video", action="store_true", help="Skip video preprocessing")
    parser.add_argument("--images-only", action="store_true", help="Only process images")
    parser.add_argument("--urls-only", action="store_true", help="Only process URLs")
    parser.add_argument("--video-only", action="store_true", help="Only process videos")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    all_reports = {}

    if args.urls_only:
        all_reports["url"] = [r.to_dict() for r in run_url_preprocessing()]
    elif args.video_only:
        all_reports["video"] = [r.to_dict() for r in run_video_preprocessing()]
    elif args.images_only:
        all_reports["image"] = [r.to_dict() for r in run_image_preprocessing()]
    else:
        all_reports["image"] = [r.to_dict() for r in run_image_preprocessing()]
        if not args.skip_video:
            all_reports["video"] = [r.to_dict() for r in run_video_preprocessing()]
        all_reports["url"] = [r.to_dict() for r in run_url_preprocessing()]

    total_time = time.time() - start

    # Save combined report
    report_path = REPORT_DIR / "preprocessing_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "total_time_seconds": round(total_time, 2),
            "reports": all_reports,
        }, f, indent=2)

    print("\n" + "=" * 60)
    print(f"✅ PREPROCESSING COMPLETE — {total_time:.1f}s total")
    print(f"   Report saved: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
