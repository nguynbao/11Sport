from __future__ import annotations

import tempfile
from pathlib import Path

from tenacity import retry

from ..config import get_settings
from ..constants import OPENAI_VOICES
from ..media.ffmpeg import to_wav48_mono
from ._common import RETRY_KW, make_client


class OpenAITTS:
    name = "openai"

    def __init__(self) -> None:
        self.client = make_client()

    def voice(self, lang: str, gender: str) -> str:
        return OPENAI_VOICES[gender]

    @retry(**RETRY_KW)
    def _speech(self, text: str, voice: str, rate: float) -> bytes:
        resp = self.client.audio.speech.create(
            model=get_settings().OPENAI_TTS_MODEL, voice=voice, input=text, speed=rate, response_format="wav"
        )
        return resp.read()

    def synthesize(self, text: str, lang: str, gender: str, rate: float, out_path: Path) -> None:
        data = self._speech(text, self.voice(lang, gender), rate)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            to_wav48_mono(tmp_path, out_path)
        finally:
            tmp_path.unlink(missing_ok=True)
