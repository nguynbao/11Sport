from __future__ import annotations

import logging
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import get_settings
from ..errors import PipelineError
from ..media import ffmpeg as F

log = logging.getLogger(__name__)

ELEVENLABS_VOICES: dict[str, str] = {
    "adam": "pNInz6obpgDQGcFmaJgB",    # Adam - Dominant, Firm
    "female": "21m00Tcm4TlvDq8ikWAM",  # Rachel
    "male": "nPczCjzI2devNBz1zQrb",    # Brian - Deep, Resonant
}

_RETRY = {
    "stop": stop_after_attempt(3),
    "wait": wait_exponential(multiplier=1, min=1, max=6),
    "retry": retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
    "reraise": True,
}


def verify_elevenlabs_key(api_key: str) -> dict:
    """Verify an ElevenLabs API key and return subscription/quota metadata if available."""
    key = api_key.strip()
    if not key:
        raise ValueError("API key trống")
    r = httpx.get(
        "https://api.elevenlabs.io/v1/user/subscription",
        headers={"xi-api-key": key},
        timeout=15.0,
    )
    if r.status_code == 200:
        data = r.json()
        return {
            "valid": True,
            "tier": data.get("tier", "free"),
            "character_count": data.get("character_count"),
            "character_limit": data.get("character_limit"),
        }
    # Some scoped API keys only allow TTS and not /v1/user/subscription; test with tiny TTS call
    r_tts = httpx.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICES['adam']}?output_format=mp3_44100_128",
        headers={"xi-api-key": key, "Content-Type": "application/json"},
        json={"text": "OK", "model_id": get_settings().ELEVENLABS_MODEL},
        timeout=20.0,
    )
    if r_tts.status_code == 200 and len(r_tts.content) > 100:
        return {"valid": True, "tier": "scoped_key", "character_count": None, "character_limit": None}
    detail = ""
    try:
        detail = r_tts.json().get("detail", {}).get("message", "") or str(r_tts.json())
    except Exception:  # noqa: BLE001
        detail = r_tts.text[:200]
    raise ValueError(f"Key ElevenLabs không hợp lệ hoặc hết hạn mức ({r_tts.status_code}): {detail}")


class ElevenLabsTTS:
    name = "elevenlabs"

    def voice(self, lang: str, gender: str) -> str:
        vid = ELEVENLABS_VOICES.get(gender, ELEVENLABS_VOICES["adam"])
        return f"elevenlabs-v25-{lang}-{gender}-{vid[:6]}"

    @retry(**_RETRY)
    def _fetch_mp3(self, text: str, lang: str, voice_id: str, gender: str, mp3_path: Path) -> None:
        s = get_settings()
        key = s.ELEVENLABS_API_KEY.strip()
        if not key:
            raise PipelineError("Chưa cấu hình ELEVENLABS_API_KEY")
        # eleven_multilingual_v2 does NOT support Vietnamese ('vi'); always use eleven_turbo_v2_5 for 'vi'
        model_id = s.ELEVENLABS_MODEL
        if lang == "vi" and model_id == "eleven_multilingual_v2":
            model_id = "eleven_turbo_v2_5"
        voice_settings = (
            {"stability": 0.45, "similarity_boost": 0.80, "style": 0.30, "use_speaker_boost": True}
            if gender == "adam"
            else {"stability": 0.50, "similarity_boost": 0.75, "style": 0.0, "use_speaker_boost": True}
        )
        payload: dict = {
            "text": text,
            "model_id": model_id,
            "voice_settings": voice_settings,
        }
        if lang in ("vi", "en") and ("2_5" in model_id or "v3" in model_id):
            payload["language_code"] = lang
        r = httpx.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128",
            headers={"xi-api-key": key, "Content-Type": "application/json"},
            json=payload,
            timeout=45.0,
        )
        if r.status_code != 200:
            detail = ""
            try:
                detail = r.json().get("detail", {}).get("message", "") or str(r.json())
            except Exception:  # noqa: BLE001
                detail = r.text[:200]
            raise RuntimeError(f"ElevenLabs HTTP {r.status_code}: {detail}")
        if len(r.content) < 100:
            raise RuntimeError("ElevenLabs trả về audio rỗng")
        mp3_path.write_bytes(r.content)

    def synthesize(self, text: str, lang: str, gender: str, rate: float, out_path: Path) -> None:
        voice_id = ELEVENLABS_VOICES.get(gender, ELEVENLABS_VOICES["adam"])
        mp3 = out_path.with_suffix(".el.mp3")
        raw_wav = out_path.with_suffix(".el_raw.wav")
        try:
            self._fetch_mp3(text, lang, voice_id, gender, mp3)
        except Exception as e:
            raise PipelineError(f"ElevenLabs TTS lỗi: {e}") from e
        try:
            if abs(rate - 1.0) > 0.02:
                F.to_wav48_mono(mp3, raw_wav)
                F.atempo(raw_wav, out_path, max(0.5, min(2.0, rate)))
            else:
                F.to_wav48_mono(mp3, out_path)
        finally:
            mp3.unlink(missing_ok=True)
            raw_wav.unlink(missing_ok=True)
