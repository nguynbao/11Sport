from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..config import BACKEND_DIR, get_settings, save_env_var
from ..constants import GENDERS, LANG_LABELS, PREVIEW_SAMPLES, TARGET_LANGS, VOICE_LABELS
from ..providers import get_tts
from ..providers.elevenlabs_tts import verify_elevenlabs_key

router = APIRouter(tags=["meta"])


def _active_tts_provider() -> str:
    s = get_settings()
    if s.PROVIDER_MODE == "fake":
        return "fake"
    if s.ELEVENLABS_API_KEY.strip() or s.TTS_PROVIDER == "elevenlabs":
        return "elevenlabs"
    return "edge" if s.PROVIDER_MODE == "free" else s.TTS_PROVIDER


def _mask_key(k: str) -> str:
    k = k.strip()
    if len(k) <= 10:
        return "********" if k else ""
    return f"{k[:6]}...{k[-4:]}"


@router.get("/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "provider_mode": s.PROVIDER_MODE,
        "tts_provider": _active_tts_provider(),
    }


@router.get("/config")
def config() -> dict:
    s = get_settings()
    has_el = bool(s.ELEVENLABS_API_KEY.strip())
    return {
        "target_languages": [{"code": c, "label": LANG_LABELS[c]} for c in TARGET_LANGS],
        "voices": [{"code": g, "label": VOICE_LABELS[g]} for g in GENDERS],
        "max_upload_mb": s.MAX_UPLOAD_MB,
        "max_duration_sec": s.MAX_DURATION_SEC,
        "default_orig_volume": s.DEFAULT_ORIG_VOLUME,
        "tts_provider": _active_tts_provider(),
        "elevenlabs_configured": has_el,
        "elevenlabs_key_masked": _mask_key(s.ELEVENLABS_API_KEY) if has_el else "",
    }


class ElevenLabsKeyIn(BaseModel):
    api_key: str = ""


@router.post("/settings/elevenlabs")
def save_elevenlabs_settings(body: ElevenLabsKeyIn) -> dict:
    key = body.api_key.strip()
    voices_dir = BACKEND_DIR / "assets" / "voices"
    if not key:
        save_env_var("ELEVENLABS_API_KEY", "")
        return {
            "configured": False,
            "tts_provider": _active_tts_provider(),
            "elevenlabs_key_masked": "",
            "message": "Đã xóa ElevenLabs API Key — chuyển về giọng đọc miễn phí (Edge Neural TTS).",
        }
    try:
        info = verify_elevenlabs_key(key)
    except Exception as e:
        raise HTTPException(400, str(e)) from e
    save_env_var("ELEVENLABS_API_KEY", key)
    if voices_dir.exists():
        for p in voices_dir.glob("*_elevenlabs.wav"):
            p.unlink(missing_ok=True)
    return {
        "configured": True,
        "tts_provider": "elevenlabs",
        "elevenlabs_key_masked": _mask_key(key),
        "message": "Đã lưu & kích hoạt ElevenLabs thành công (giọng Adam Dominant, Firm chính gốc)!",
        **info,
    }


@router.get("/voices/preview")
def preview_voice(voice: str = "adam", lang: str = "en") -> FileResponse:
    if voice not in GENDERS:
        raise HTTPException(400, "Giọng đọc không hợp lệ")
    if lang not in TARGET_LANGS:
        raise HTTPException(400, "Ngôn ngữ không hợp lệ")
    voices_dir = BACKEND_DIR / "assets" / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    tts = get_tts(voice)
    suffix = tts.name
    out = voices_dir / f"{voice}_{lang}_{suffix}.wav"
    if not out.exists():
        tmp = voices_dir / f"{voice}_{lang}_{suffix}.tmp.wav"
        tts.synthesize(PREVIEW_SAMPLES[lang], lang, voice, 1.0, tmp)
        tmp.rename(out)
    return FileResponse(
        out,
        media_type="audio/wav",
        filename=f"{voice}_{lang}.wav",
        headers={"Cache-Control": "no-cache"},
    )

