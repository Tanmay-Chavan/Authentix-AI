"""Video frame extraction pipeline using FFmpeg.

Extracts frames from video files for downstream deepfake detection.
Uses subprocess guard for secure FFmpeg invocation.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Optional

import structlog

from authentix.security.guards import validate_command

logger = structlog.get_logger(__name__)


async def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: float = 1.0,
    max_frames: int = 30,
    size: Optional[tuple[int, int]] = (380, 380),
    quality: int = 2,
) -> list[Path]:
    """Extract frames from a video using FFmpeg.

    Parameters
    ----------
    video_path : str | Path
        Path to the input video file.
    output_dir : str | Path
        Directory to save extracted frames.
    fps : float
        Frames per second to extract. 1.0 = one frame per second.
    max_frames : int
        Maximum number of frames to extract.
    size : tuple[int, int], optional
        Resize frames to (width, height). None = keep original.
    quality : int
        JPEG quality (2 = high, 31 = low). Lower is better.

    Returns
    -------
    list[Path]
        Sorted list of paths to extracted frame images.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Build FFmpeg command
    output_pattern = str(output_dir / "frame_%04d.jpg")

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", _build_filter_chain(fps, size),
        "-frames:v", str(max_frames),
        "-qscale:v", str(quality),
        "-y",  # Overwrite existing
        output_pattern,
    ]

    # Security: validate command through subprocess guard
    validate_command(cmd)

    # Run FFmpeg asynchronously
    logger.info(
        "video.extracting_frames",
        video=str(video_path),
        fps=fps,
        max_frames=max_frames,
        size=size,
    )

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()

    if process.returncode != 0:
        error_msg = stderr.decode("utf-8", errors="replace")
        logger.error(
            "video.ffmpeg_error",
            video=str(video_path),
            returncode=process.returncode,
            stderr=error_msg[:500],
        )
        raise RuntimeError(
            f"FFmpeg failed (code {process.returncode}): {error_msg[:200]}"
        )

    # Collect output frames
    frames = sorted(output_dir.glob("frame_*.jpg"))
    logger.info(
        "video.frames_extracted",
        video=str(video_path),
        frame_count=len(frames),
    )

    return frames


def extract_frames_sync(
    video_path: str | Path,
    output_dir: str | Path,
    fps: float = 1.0,
    max_frames: int = 30,
    size: Optional[tuple[int, int]] = (380, 380),
    quality: int = 2,
) -> list[Path]:
    """Synchronous version of frame extraction (for training scripts).

    Parameters are identical to ``extract_frames``.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_pattern = str(output_dir / "frame_%04d.jpg")

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", _build_filter_chain(fps, size),
        "-frames:v", str(max_frames),
        "-qscale:v", str(quality),
        "-y",
        output_pattern,
    ]

    validate_command(cmd)

    result = subprocess.run(
        cmd,
        capture_output=True,
        timeout=120,
    )

    if result.returncode != 0:
        error_msg = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"FFmpeg failed (code {result.returncode}): {error_msg[:200]}"
        )

    frames = sorted(output_dir.glob("frame_*.jpg"))
    return frames


async def get_video_info(video_path: str | Path) -> dict:
    """Get video metadata using FFprobe.

    Returns
    -------
    dict
        Contains: duration, width, height, fps, codec, total_frames.
    """
    video_path = Path(video_path)
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    validate_command(cmd)

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await process.communicate()

    import json
    data = json.loads(stdout.decode("utf-8"))

    # Find video stream
    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        raise ValueError(f"No video stream found in {video_path}")

    # Parse FPS from r_frame_rate (e.g., "30/1")
    fps_parts = video_stream.get("r_frame_rate", "30/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0

    return {
        "duration": float(data.get("format", {}).get("duration", 0)),
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "fps": fps,
        "codec": video_stream.get("codec_name", "unknown"),
        "total_frames": int(video_stream.get("nb_frames", 0)),
    }


def _build_filter_chain(
    fps: float,
    size: Optional[tuple[int, int]],
) -> str:
    """Build FFmpeg video filter chain string."""
    filters = [f"fps={fps}"]
    if size is not None:
        w, h = size
        filters.append(f"scale={w}:{h}")
    return ",".join(filters)
