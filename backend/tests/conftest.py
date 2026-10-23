from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

LOCAL_BIN = Path.home() / ".local" / "bin"
if (LOCAL_BIN / "ffmpeg").exists():
    os.environ.setdefault("FFMPEG_BIN", str(LOCAL_BIN / "ffmpeg"))
    os.environ.setdefault("FFPROBE_BIN", str(LOCAL_BIN / "ffprobe"))
os.environ["PROVIDER_MODE"] = "fake"


def ffmpeg() -> str:
    return os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Isolated storage + DB in fake provider mode."""
    from app import db as dbmod
    from app.config import get_settings

    monkeypatch.setenv("PROVIDER_MODE", "fake")
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'app.db'}")
    get_settings.cache_clear()
    dbmod.reset_engine()
    dbmod.init_db()
    yield tmp_path
    dbmod.reset_engine()
    get_settings.cache_clear()


def make_video(path: Path, seconds: float = 10.0, size: str = "640x360", codec: str = "libx264",
               audio: bool = True) -> Path:
    cmd = [ffmpeg(), "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=25:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    cmd += ["-c:v", codec, "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac" if path.suffix != ".webm" else "libopus", "-shortest"]
    cmd.append(str(path))
    subprocess.run(cmd, check=True, capture_output=True)
    return path


@pytest.fixture()
def sample_video(tmp_path) -> Path:
    return make_video(tmp_path / "sample.mp4")
