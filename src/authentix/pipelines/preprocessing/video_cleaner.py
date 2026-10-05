"""Phase 1.5 — Video Data Preprocessing & Cleaning Pipeline.

Implements the 4-step video preprocessing pipeline from
the review-2 defense document:

1. Container:  ffprobe validation — drops corrupted video streams
2. Sampling:   Temporal extraction — 1 frame/15 (~2fps), max 30 frames
3. Extraction: Face crop with spatial margin (1.3× bbox) [optional MTCNN]
4. Outliers:   Laplacian variance blur filter (σ² ≥ 100)

Falls back to center-crop if no face detector is available.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import structlog

from authentix.security.guards import validate_command

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

FRAME_SAMPLE_EVERY_N = 15      # Extract 1 frame every 15 (~2fps for 30fps video)
MAX_FRAMES = 30                 # Maximum frames to extract per video
LAPLACIAN_VARIANCE_MIN = 100.0  # Minimum blur threshold
FACE_MARGIN = 1.3               # Spatial margin multiplier for face crops
MIN_FACE_SIZE = 64              # Minimum face crop dimension in pixels
OUTPUT_SIZE = (380, 380)        # EfficientNet-B4 input size


@dataclass
class VideoCleaningReport:
    """Report generated after cleaning a video dataset."""

    dataset_name: str
    total_videos: int = 0
    valid_videos: int = 0
    corrupted_videos: int = 0
    total_frames_extracted: int = 0
    blurry_frames_dropped: int = 0
    duration_seconds: float = 0.0
    errors: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════
# Step 1: Container Validation (ffprobe)
# ═══════════════════════════════════════════════════════════════

def validate_video_container(video_path: Path) -> tuple[bool, dict]:
    """Validate video container using ffprobe.

    Returns (is_valid, metadata_dict).
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    try:
        validate_command(cmd)
        result = subprocess.run(
            cmd, capture_output=True, timeout=30
        )

        if result.returncode != 0:
            return False, {"error": "ffprobe failed"}

        data = json.loads(result.stdout.decode("utf-8"))

        # Find video stream
        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if video_stream is None:
            return False, {"error": "No video stream found"}

        metadata = {
            "duration": float(data.get("format", {}).get("duration", 0)),
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "codec": video_stream.get("codec_name", "unknown"),
            "fps": _parse_fps(video_stream.get("r_frame_rate", "30/1")),
        }

        return True, metadata

    except Exception as e:
        return False, {"error": str(e)[:200]}


# ═══════════════════════════════════════════════════════════════
# Step 2: Temporal Frame Sampling
# ═══════════════════════════════════════════════════════════════

def extract_sampled_frames(
    video_path: Path,
    output_dir: Path,
    sample_every_n: int = FRAME_SAMPLE_EVERY_N,
    max_frames: int = MAX_FRAMES,
) -> list[Path]:
    """Extract temporally sampled frames from a video.

    Extracts 1 frame every `sample_every_n` frames (≈2fps for 30fps video).
    Caps at `max_frames` total frames.

    Returns list of saved frame paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_paths = []
    frame_idx = 0
    saved_count = 0

    while saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every_n == 0:
            frame_path = output_dir / f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_paths.append(frame_path)
            saved_count += 1

        frame_idx += 1

    cap.release()
    return frame_paths


# ═══════════════════════════════════════════════════════════════
# Step 3: Face Cropping (Center-Crop Fallback)
# ═══════════════════════════════════════════════════════════════

def crop_face_region(
    frame_path: Path,
    output_path: Path,
    margin: float = FACE_MARGIN,
    output_size: tuple[int, int] = OUTPUT_SIZE,
) -> bool:
    """Crop the face region from a frame.

    Uses OpenCV's Haar Cascade face detector as a lightweight alternative
    to MTCNN. Falls back to center-crop if no face is found.

    Returns True if a face was detected, False if center-cropped.
    """
    img = cv2.imread(str(frame_path))
    if img is None:
        return False

    h, w = img.shape[:2]

    # Try face detection with Haar Cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
    )

    if len(faces) > 0:
        # Take the largest face
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])

        # Apply margin
        cx, cy = fx + fw // 2, fy + fh // 2
        side = int(max(fw, fh) * margin)
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)

        crop = img[y1:y2, x1:x2]
        face_found = True
    else:
        # Fallback: center crop
        crop_size = min(h, w)
        y1 = (h - crop_size) // 2
        x1 = (w - crop_size) // 2
        crop = img[y1:y1 + crop_size, x1:x1 + crop_size]
        face_found = False

    # Resize to output size
    resized = cv2.resize(crop, output_size, interpolation=cv2.INTER_LANCZOS4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return face_found


# ═══════════════════════════════════════════════════════════════
# Step 4: Blur Detection (Laplacian Variance)
# ═══════════════════════════════════════════════════════════════

def compute_blur_score(image_path: Path) -> float:
    """Compute Laplacian variance as a blur metric.

    Higher value = sharper image. Values below LAPLACIAN_VARIANCE_MIN
    indicate excessively blurry frames that confuse the optimizer.

    Returns the Laplacian variance (σ²).
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def filter_blurry_frames(
    frame_paths: list[Path],
    min_variance: float = LAPLACIAN_VARIANCE_MIN,
) -> tuple[list[Path], list[Path]]:
    """Filter out blurry frames based on Laplacian variance.

    Returns (sharp_frames, blurry_frames).
    """
    sharp = []
    blurry = []

    for path in frame_paths:
        score = compute_blur_score(path)
        if score >= min_variance:
            sharp.append(path)
        else:
            blurry.append(path)

    return sharp, blurry


# ═══════════════════════════════════════════════════════════════
# Full Video Pipeline
# ═══════════════════════════════════════════════════════════════

def preprocess_video_dataset(
    video_dir: Path,
    output_dir: Path,
    dataset_name: str,
    with_face_crop: bool = True,
) -> VideoCleaningReport:
    """Run the full 4-step video preprocessing pipeline.

    Processes all .mp4 files in video_dir and outputs cleaned
    face-cropped frames to output_dir.
    """
    start_time = time.time()
    report = VideoCleaningReport(dataset_name=dataset_name)

    videos = sorted(video_dir.glob("*.mp4"))
    report.total_videos = len(videos)

    logger.info(
        "video_preprocessing.start",
        dataset=dataset_name,
        total_videos=len(videos),
    )

    for video_path in videos:
        # Step 1: Validate container
        is_valid, metadata = validate_video_container(video_path)
        if not is_valid:
            report.corrupted_videos += 1
            report.errors.append({
                "video": video_path.name,
                "step": "container",
                "error": metadata.get("error", "unknown"),
            })
            continue

        report.valid_videos += 1
        video_output = output_dir / video_path.stem

        # Step 2: Extract sampled frames
        try:
            raw_frames = extract_sampled_frames(video_path, video_output / "raw")
        except Exception as e:
            report.errors.append({
                "video": video_path.name,
                "step": "sampling",
                "error": str(e)[:200],
            })
            continue

        # Step 3: Face crop (or center crop fallback)
        cropped_frames = []
        if with_face_crop:
            for frame_path in raw_frames:
                cropped_path = video_output / "cropped" / frame_path.name
                crop_face_region(frame_path, cropped_path)
                cropped_frames.append(cropped_path)
        else:
            cropped_frames = raw_frames

        # Step 4: Blur filter
        sharp_frames, blurry_frames = filter_blurry_frames(cropped_frames)
        report.total_frames_extracted += len(sharp_frames)
        report.blurry_frames_dropped += len(blurry_frames)

    report.duration_seconds = round(time.time() - start_time, 2)

    logger.info(
        "video_preprocessing.complete",
        dataset=dataset_name,
        valid_videos=report.valid_videos,
        corrupted=report.corrupted_videos,
        total_frames=report.total_frames_extracted,
        blurry_dropped=report.blurry_frames_dropped,
        duration=f"{report.duration_seconds:.1f}s",
    )

    return report


# ═══════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════

def _parse_fps(fps_str: str) -> float:
    """Parse FFprobe's r_frame_rate string (e.g., '30/1' → 30.0)."""
    parts = fps_str.split("/")
    if len(parts) == 2:
        return float(parts[0]) / max(float(parts[1]), 1)
    return 30.0
