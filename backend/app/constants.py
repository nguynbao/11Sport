from __future__ import annotations

from enum import StrEnum


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    PREPARING = "PREPARING"
    TRANSCRIBING = "TRANSCRIBING"
    TRANSLATING = "TRANSLATING"
    REVIEW = "REVIEW"
    RENDER_QUEUED = "RENDER_QUEUED"
    SYNTHESIZING = "SYNTHESIZING"
    MIXING = "MIXING"
    RENDERING = "RENDERING"
    DONE = "DONE"
    FAILED = "FAILED"


PHASE_A_RUNNING = {JobStatus.PREPARING, JobStatus.TRANSCRIBING, JobStatus.TRANSLATING}
PHASE_B_RUNNING = {JobStatus.SYNTHESIZING, JobStatus.MIXING, JobStatus.RENDERING}
RUNNING_STATUSES = PHASE_A_RUNNING | PHASE_B_RUNNING

TARGET_LANGS = ("en", "vi")
# "zh" kept in the lookup tables below only so legacy jobs created before the switch still render.
LANG_LABELS = {"en": "Tiếng Anh", "vi": "Tiếng Việt", "zh": "Tiếng Trung (giản thể)"}
LANG_SHORT_VI = {"en": "Tiếng Anh", "vi": "Tiếng Việt", "zh": "Tiếng Trung"}
XML_LANG = {"en": "en-US", "vi": "vi-VN", "zh": "zh-CN"}
ISO639_2 = {"en": "eng", "vi": "vie", "zh": "chi"}

# Whisper may report the language as a name ("vietnamese") or an ISO code ("vi").
LANG_ALIASES = {"english": "en", "vietnamese": "vi", "chinese": "zh", "mandarin": "zh"}


def normalize_lang(code: str | None) -> str:
    c = (code or "").strip().lower()
    return LANG_ALIASES.get(c, c.split("-")[0])


GENDERS = ("adam", "female", "male")
VOICE_LABELS = {
    "adam": "Adam (Dominant, Firm)",
    "female": "Nữ",
    "male": "Nam",
}
VOICE_PITCH = {
    "adam": "-6Hz",
    "female": "+0Hz",
    "male": "+0Hz",
}
SUBTITLE_MODES = ("hard", "soft", "none")
ALLOWED_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}

# Same neural voice names work for both Azure Speech (paid) and Edge TTS (free demo).
AZURE_VOICES = {
    ("en", "adam"): "en-US-ChristopherNeural",
    ("en", "female"): "en-US-JennyNeural",
    ("en", "male"): "en-US-GuyNeural",
    ("vi", "adam"): "vi-VN-NamMinhNeural",
    ("vi", "female"): "vi-VN-HoaiMyNeural",
    ("vi", "male"): "vi-VN-NamMinhNeural",
    ("zh", "adam"): "zh-CN-YunxiNeural",
    ("zh", "female"): "zh-CN-XiaoxiaoNeural",
    ("zh", "male"): "zh-CN-YunxiNeural",
}
OPENAI_VOICES = {"adam": "onyx", "female": "nova", "male": "echo"}

PREVIEW_SAMPLES = {
    "en": "Hello everyone, welcome to 11 Sport!",
    "vi": "Xin chào các bạn đã đến với 11 Sport!",
}

SAMPLE_RATE = 48000
