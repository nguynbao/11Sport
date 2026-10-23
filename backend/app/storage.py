from __future__ import annotations

import shutil
from pathlib import Path

from .config import get_settings


def storage_root() -> Path:
    root = get_settings().storage_path
    root.mkdir(parents=True, exist_ok=True)
    return root


def job_dir(job_id: str) -> Path:
    d = storage_root() / "jobs" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def tts_dir(job_id: str) -> Path:
    d = job_dir(job_id) / "tts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def asr_cache_dir() -> Path:
    d = storage_root() / "cache" / "asr"
    d.mkdir(parents=True, exist_ok=True)
    return d


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(storage_root().resolve()))


def abs_(relpath: str) -> Path:
    return storage_root() / relpath


def delete_job_dir(job_id: str) -> None:
    d = storage_root() / "jobs" / job_id
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
