from __future__ import annotations

from ..config import get_settings
from .base import ASRProvider, Translator, TTSProvider


def get_asr() -> ASRProvider:
    mode = get_settings().PROVIDER_MODE
    if mode == "fake":
        from .fake import FakeASR
        return FakeASR()
    if mode == "free":
        from .free import LocalWhisperASR
        return LocalWhisperASR()
    from .openai_asr import OpenAIASR
    return OpenAIASR()


def get_translator() -> Translator:
    mode = get_settings().PROVIDER_MODE
    if mode == "fake":
        from .fake import FakeTranslator
        return FakeTranslator()
    if mode == "free":
        from .free import FreeLLMTranslator
        return FreeLLMTranslator()
    from .openai_llm import OpenAITranslator
    return OpenAITranslator()


def get_tts(gender: str | None = None) -> TTSProvider:
    s = get_settings()
    if s.PROVIDER_MODE == "fake":
        from .fake import FakeTTS
        return FakeTTS()
    if gender == "adam" and (s.ELEVENLABS_API_KEY.strip() or s.TTS_PROVIDER == "elevenlabs"):
        from .elevenlabs_tts import ElevenLabsTTS
        return ElevenLabsTTS()
    if s.PROVIDER_MODE == "free":
        from .free import EdgeFreeTTS
        return EdgeFreeTTS()
    if s.TTS_PROVIDER == "elevenlabs":
        from .elevenlabs_tts import ElevenLabsTTS
        return ElevenLabsTTS()
    if s.TTS_PROVIDER == "openai":
        from .openai_tts import OpenAITTS
        return OpenAITTS()
    from .azure_tts import AzureTTS
    return AzureTTS()


def tts_voice_name(tts: TTSProvider, lang: str, gender: str) -> str:
    fn = getattr(tts, "voice", None)
    return fn(lang, gender) if fn else f"{tts.name}-{lang}-{gender}"
