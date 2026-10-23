from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _resolve(p: str) -> Path:
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = (BACKEND_DIR / path).resolve()
    return path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(BACKEND_DIR / ".env"), extra="ignore")

    PROVIDER_MODE: str = "real"
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""
    OPENAI_ASR_MODEL: str = "whisper-1"
    OPENAI_LLM_MODEL: str = "gpt-4o-mini"
    OPENAI_TTS_MODEL: str = "tts-1"
    TTS_PROVIDER: str = "azure"
    # PROVIDER_MODE=free (demo, no API key): local faster-whisper + Google Translate + Edge TTS
    WHISPER_MODEL: str = "medium"         # tiny | base | small | medium | large-v3 (larger = slower, better)
    WHISPER_COMPUTE_TYPE: str = "int8"
    AZURE_SPEECH_KEY: str = ""
    AZURE_SPEECH_REGION: str = "southeastasia"
    ELEVENLABS_API_KEY: str = ""
    ELEVENLABS_MODEL: str = "eleven_turbo_v2_5"
    DATABASE_URL: str = "sqlite:///./data/app.db"
    STORAGE_DIR: str = "../storage"
    FONTS_DIR: str = "./assets/fonts"
    FFMPEG_BIN: str = "ffmpeg"
    FFPROBE_BIN: str = "ffprobe"
    MAX_UPLOAD_MB: int = 500
    MAX_DURATION_SEC: int = 600
    RETENTION_DAYS: int = 14
    CORS_ORIGINS: str = "http://localhost:5173"
    DEFAULT_ORIG_VOLUME: float = 0.05
    WORKER_POLL_SEC: float = 1.0

    @property
    def storage_path(self) -> Path:
        return _resolve(self.STORAGE_DIR)

    @property
    def fonts_path(self) -> Path:
        return _resolve(self.FONTS_DIR)

    @property
    def database_url(self) -> str:
        """Resolve relative sqlite paths against the backend dir so cwd does not matter."""
        prefix = "sqlite:///"
        if self.DATABASE_URL.startswith(prefix) and not self.DATABASE_URL.startswith(prefix + "/"):
            rel = self.DATABASE_URL[len(prefix):]
            return prefix + str(_resolve(rel))
        return self.DATABASE_URL

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    def validate_runtime(self) -> None:
        if self.PROVIDER_MODE not in ("real", "free", "fake"):
            raise RuntimeError("PROVIDER_MODE phải là 'real', 'free' hoặc 'fake'")
        if self.TTS_PROVIDER not in ("azure", "openai", "elevenlabs"):
            raise RuntimeError("TTS_PROVIDER phải là 'azure', 'openai' hoặc 'elevenlabs'")
        if self.PROVIDER_MODE == "real":
            if not self.OPENAI_API_KEY:
                raise RuntimeError("Thiếu OPENAI_API_KEY trong backend/.env (hoặc đặt PROVIDER_MODE=fake)")
            if self.TTS_PROVIDER == "azure" and not self.AZURE_SPEECH_KEY:
                raise RuntimeError("Thiếu AZURE_SPEECH_KEY (hoặc đặt TTS_PROVIDER=openai)")


_env_mtime: float = -1.0


@lru_cache
def _cached_settings() -> Settings:
    return Settings()


def get_settings() -> Settings:
    global _env_mtime
    env_file = BACKEND_DIR / ".env"
    try:
        mtime = env_file.stat().st_mtime if env_file.exists() else 0.0
    except OSError:
        mtime = 0.0
    if mtime != _env_mtime:
        _cached_settings.cache_clear()
        _env_mtime = mtime
    return _cached_settings()


get_settings.cache_clear = _cached_settings.cache_clear  # type: ignore[attr-defined]


def save_env_var(key: str, value: str) -> None:
    """Persist a key=value setting into backend/.env and invalidate settings cache."""
    env_file = BACKEND_DIR / ".env"
    lines = env_file.read_text("utf-8").splitlines() if env_file.exists() else []
    updated = False
    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            updated = True
        else:
            new_lines.append(line)
    if not updated:
        new_lines.append(f"{key}={value}")
    env_file.write_text("\n".join(new_lines) + "\n", "utf-8")
    _cached_settings.cache_clear()

