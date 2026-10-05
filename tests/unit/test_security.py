"""Gate 1 — Tests for security guards (upload, SSRF, subprocess)."""

import pytest
from pathlib import Path
import tempfile

from authentix.security.guards import (
    UploadSanitiserError,
    SSRFError,
    SubprocessGuardError,
    sanitise_filename,
    validate_upload,
    validate_url,
    validate_command,
)


class TestSanitiseFilename:
    """Verify filename sanitisation strips dangerous patterns."""

    def test_normal_filename(self):
        result = sanitise_filename("photo.jpg")
        assert result.endswith(".jpg")
        assert len(result) == 36  # 32 hex chars + ".jpg"

    def test_path_traversal(self):
        result = sanitise_filename("../../etc/passwd.jpg")
        assert ".." not in result
        assert result.endswith(".jpg")

    def test_null_bytes(self):
        result = sanitise_filename("malicious\x00.jpg")
        assert "\x00" not in result

    def test_extension_sanitisation(self):
        result = sanitise_filename("file.<script>.jpg")
        # Only alphanumeric + dot allowed in extension
        assert "<" not in result
        assert ">" not in result


class TestUploadValidation:
    """Verify upload sanitiser rejects bad files."""

    def test_empty_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(UploadSanitiserError, match="Empty file"):
                validate_upload(b"", "test.jpg", Path(tmpdir))

    def test_oversized_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            huge = b"\xff" * (21 * 1024 * 1024)  # 21 MB
            with pytest.raises(UploadSanitiserError, match="too large"):
                validate_upload(huge, "big.jpg", Path(tmpdir))

    def test_bad_extension_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(UploadSanitiserError, match="Disallowed"):
                validate_upload(b"\xff\xd8\xff", "shell.exe", Path(tmpdir))

    def test_bad_magic_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Valid extension but wrong magic bytes
            with pytest.raises(UploadSanitiserError, match="header"):
                validate_upload(b"not-a-jpeg", "fake.jpg", Path(tmpdir))

    def test_valid_jpeg_accepted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Minimal JPEG header
            jpeg_header = b"\xff\xd8\xff\xe0" + b"\x00" * 100
            path = validate_upload(jpeg_header, "photo.jpg", Path(tmpdir))
            assert path.exists()
            assert path.suffix == ".jpg"

    def test_valid_png_accepted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # PNG header
            png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
            path = validate_upload(png_header, "image.png", Path(tmpdir))
            assert path.exists()
            assert path.suffix == ".png"


class TestSSRFGuard:
    """Verify SSRF guard blocks internal URLs."""

    def test_localhost_blocked(self):
        with pytest.raises(SSRFError, match="blocked IP"):
            validate_url("http://127.0.0.1/admin")

    def test_private_ip_blocked(self):
        with pytest.raises(SSRFError, match="blocked IP"):
            validate_url("http://192.168.1.1/secret")

    def test_file_scheme_blocked(self):
        with pytest.raises(SSRFError, match="Blocked URL scheme"):
            validate_url("file:///etc/passwd")

    def test_no_scheme_rejected(self):
        with pytest.raises(SSRFError, match="no scheme"):
            validate_url("example.com")

    def test_ftp_blocked(self):
        with pytest.raises(SSRFError, match="Blocked URL scheme"):
            validate_url("ftp://evil.com/file")

    def test_too_long_url_rejected(self):
        with pytest.raises(SSRFError, match="too long"):
            validate_url("https://example.com/" + "a" * 2100)

    def test_valid_https_accepted(self):
        # This may fail if DNS doesn't resolve — skip in CI
        try:
            result = validate_url("https://www.google.com")
            assert result == "https://www.google.com"
        except SSRFError:
            pytest.skip("DNS resolution unavailable")


class TestSubprocessGuard:
    """Verify subprocess guard enforces command whitelist."""

    def test_ffmpeg_allowed(self):
        cmd = validate_command(["ffmpeg", "-i", "input.mp4", "output.mp4"])
        assert cmd[0] == "ffmpeg"

    def test_ffprobe_allowed(self):
        cmd = validate_command(["ffprobe", "-v", "quiet", "test.mp4"])
        assert cmd[0] == "ffprobe"

    def test_python_blocked(self):
        with pytest.raises(SubprocessGuardError, match="not in whitelist"):
            validate_command(["python", "-c", "import os; os.system('rm -rf /')"])

    def test_shell_injection_blocked(self):
        with pytest.raises(SubprocessGuardError, match="Dangerous"):
            validate_command(["ffmpeg", "-i", "input.mp4; rm -rf /"])

    def test_pipe_injection_blocked(self):
        with pytest.raises(SubprocessGuardError, match="Dangerous"):
            validate_command(["ffmpeg", "-i", "input.mp4 | cat /etc/passwd"])

    def test_empty_command_blocked(self):
        with pytest.raises(SubprocessGuardError, match="Empty"):
            validate_command([])
