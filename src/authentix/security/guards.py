"""Security guards — upload sanitiser, subprocess guard, SSRF guard.

These are the first line of defence.  Every file upload and every
outbound network call MUST pass through these guards.
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import secrets
import socket
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import structlog

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# UPLOAD SANITISER
# ═══════════════════════════════════════════════════════════════

# Allowed MIME magic bytes (first N bytes of file)
MAGIC_BYTES = {
    "image/jpeg": [b"\xff\xd8\xff"],
    "image/png": [b"\x89PNG\r\n\x1a\n"],
    "image/gif": [b"GIF87a", b"GIF89a"],
    "image/webp": [b"RIFF"],  # RIFF....WEBP
    "video/mp4": [b"\x00\x00\x00", b"ftyp"],  # Various MP4 boxes
    "video/avi": [b"RIFF"],
    "video/quicktime": [b"\x00\x00\x00"],
}

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

# Max file sizes
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024   # 20 MB
MAX_VIDEO_SIZE_BYTES = 200 * 1024 * 1024  # 200 MB


class UploadSanitiserError(Exception):
    """Raised when a file fails sanitisation."""
    pass


def sanitise_filename(original_name: str) -> str:
    """Generate a safe filename: random hex + original extension.

    Strips path traversal, null bytes, and non-alphanumeric characters
    from the extension.
    """
    # Remove any path components
    basename = os.path.basename(original_name)
    # Remove null bytes
    basename = basename.replace("\x00", "")
    # Extract extension
    _, ext = os.path.splitext(basename)
    # Sanitise extension: only allow alphanumeric + dot
    ext = re.sub(r"[^a-zA-Z0-9.]", "", ext).lower()
    if ext and not ext.startswith("."):
        ext = f".{ext}"
    # Random filename
    safe_name = f"{secrets.token_hex(16)}{ext}"
    return safe_name


def validate_upload(
    file_content: bytes,
    original_filename: str,
    upload_dir: Path,
    is_video: bool = False,
) -> Path:
    """Validate and save an uploaded file securely.

    Parameters
    ----------
    file_content : bytes
        Raw file bytes.
    original_filename : str
        Original filename from the upload.
    upload_dir : Path
        Directory to save the sanitised file.
    is_video : bool
        If True, uses video size limits and extensions.

    Returns
    -------
    Path
        Absolute path to the saved, sanitised file.

    Raises
    ------
    UploadSanitiserError
        If the file fails any validation check.
    """
    max_size = MAX_VIDEO_SIZE_BYTES if is_video else MAX_IMAGE_SIZE_BYTES
    allowed_exts = ALLOWED_VIDEO_EXTENSIONS if is_video else ALLOWED_IMAGE_EXTENSIONS

    # 1. Size check
    if len(file_content) > max_size:
        raise UploadSanitiserError(
            f"File too large: {len(file_content)} bytes "
            f"(max: {max_size} bytes)"
        )

    if len(file_content) == 0:
        raise UploadSanitiserError("Empty file")

    # 2. Extension check
    _, ext = os.path.splitext(original_filename)
    ext = ext.lower()
    if ext not in allowed_exts:
        raise UploadSanitiserError(
            f"Disallowed extension: '{ext}'. "
            f"Allowed: {allowed_exts}"
        )

    # 3. Magic bytes check (basic — not a full MIME sniffer)
    _check_magic_bytes(file_content, is_video)

    # 4. Save with sanitised filename
    safe_name = sanitise_filename(original_filename)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / safe_name

    # Ensure no path traversal in the final path
    dest = dest.resolve()
    if not str(dest).startswith(str(upload_dir.resolve())):
        raise UploadSanitiserError("Path traversal detected")

    dest.write_bytes(file_content)

    # 5. Compute SHA-256 for audit trail
    sha256 = hashlib.sha256(file_content).hexdigest()
    logger.info(
        "upload.sanitised",
        original_name=original_filename,
        safe_name=safe_name,
        size_bytes=len(file_content),
        sha256=sha256,
    )

    return dest


def _check_magic_bytes(content: bytes, is_video: bool) -> None:
    """Verify file starts with expected magic bytes."""
    header = content[:16]
    if is_video:
        # MP4 / AVI / MOV — more permissive
        if not (
            header[4:8] == b"ftyp"
            or header[:4] == b"RIFF"
            or header[:3] == b"\x00\x00\x00"
        ):
            raise UploadSanitiserError(
                "File header does not match any known video format"
            )
    else:
        # Images — stricter
        if not (
            header[:3] == b"\xff\xd8\xff"        # JPEG
            or header[:8] == b"\x89PNG\r\n\x1a\n"  # PNG
            or header[:6] in (b"GIF87a", b"GIF89a")  # GIF
            or (header[:4] == b"RIFF" and content[8:12] == b"WEBP")  # WebP
            or header[:2] == b"BM"                 # BMP
        ):
            raise UploadSanitiserError(
                "File header does not match any known image format"
            )


# ═══════════════════════════════════════════════════════════════
# SSRF GUARD — Prevents Server-Side Request Forgery
# ═══════════════════════════════════════════════════════════════

# Private / reserved IP ranges that MUST be blocked
BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),       # Loopback
    ipaddress.ip_network("10.0.0.0/8"),         # Private
    ipaddress.ip_network("172.16.0.0/12"),      # Private
    ipaddress.ip_network("192.168.0.0/16"),     # Private
    ipaddress.ip_network("169.254.0.0/16"),     # Link-local
    ipaddress.ip_network("0.0.0.0/8"),          # "This" network
    ipaddress.ip_network("100.64.0.0/10"),      # Shared address space
    ipaddress.ip_network("::1/128"),            # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),           # IPv6 unique local
    ipaddress.ip_network("fe80::/10"),          # IPv6 link-local
]

# Blocked URL schemes
BLOCKED_SCHEMES = {"file", "ftp", "gopher", "data", "javascript"}

# Max URL length
MAX_URL_LENGTH = 2048


class SSRFError(Exception):
    """Raised when a URL fails SSRF checks."""
    pass


def validate_url(url: str) -> str:
    """Validate a URL is safe to fetch (no SSRF).

    Parameters
    ----------
    url : str
        The URL to validate.

    Returns
    -------
    str
        The validated URL (stripped of whitespace).

    Raises
    ------
    SSRFError
        If the URL targets internal networks or uses blocked schemes.
    """
    url = url.strip()

    # Length check
    if len(url) > MAX_URL_LENGTH:
        raise SSRFError(f"URL too long: {len(url)} chars (max {MAX_URL_LENGTH})")

    # Parse
    parsed = urlparse(url)

    # Scheme check
    if not parsed.scheme:
        raise SSRFError("URL has no scheme (missing http:// or https://)")
    if parsed.scheme.lower() in BLOCKED_SCHEMES:
        raise SSRFError(f"Blocked URL scheme: {parsed.scheme}")
    if parsed.scheme.lower() not in ("http", "https"):
        raise SSRFError(f"Only HTTP(S) URLs allowed, got: {parsed.scheme}")

    # Hostname check
    hostname = parsed.hostname
    if not hostname:
        raise SSRFError("URL has no hostname")

    # Resolve hostname to IP and check against blocked ranges
    try:
        ip_str = socket.gethostbyname(hostname)
        ip = ipaddress.ip_address(ip_str)
    except (socket.gaierror, ValueError):
        raise SSRFError(f"Cannot resolve hostname: {hostname}")

    for network in BLOCKED_NETWORKS:
        if ip in network:
            raise SSRFError(
                f"URL resolves to blocked IP range: {ip} ({network})"
            )

    logger.info("ssrf.url_validated", url=url, resolved_ip=ip_str)
    return url


# ═══════════════════════════════════════════════════════════════
# SUBPROCESS GUARD — Safe external tool execution
# ═══════════════════════════════════════════════════════════════

# Allowed external commands (whitelist)
ALLOWED_COMMANDS = {"ffmpeg", "ffprobe", "exiftool"}


class SubprocessGuardError(Exception):
    """Raised when a subprocess call is blocked."""
    pass


def validate_command(
    command: list[str],
    allowed: Optional[set[str]] = None,
) -> list[str]:
    """Validate a subprocess command against the whitelist.

    Parameters
    ----------
    command : list[str]
        The command and arguments as a list.
    allowed : set[str], optional
        Override the default allowed commands set.

    Returns
    -------
    list[str]
        The validated command list.

    Raises
    ------
    SubprocessGuardError
        If the command is not in the whitelist or contains
        shell injection characters.
    """
    if not command:
        raise SubprocessGuardError("Empty command")

    allowed_set = allowed or ALLOWED_COMMANDS
    executable = os.path.basename(command[0]).lower()

    # Remove .exe suffix on Windows
    if executable.endswith(".exe"):
        executable = executable[:-4]

    if executable not in allowed_set:
        raise SubprocessGuardError(
            f"Command '{executable}' not in whitelist: {allowed_set}"
        )

    # Check for shell injection in arguments
    dangerous_chars = {";", "|", "&", "`", "$", "(", ")", "{", "}", "<", ">"}
    for i, arg in enumerate(command[1:], 1):
        if any(c in arg for c in dangerous_chars):
            raise SubprocessGuardError(
                f"Dangerous character in argument {i}: '{arg}'"
            )

    logger.info("subprocess.validated", command=executable, argc=len(command))
    return command
