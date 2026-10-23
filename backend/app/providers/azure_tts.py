from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import httpx
from tenacity import retry

from ..config import get_settings
from ..constants import AZURE_VOICES, XML_LANG
from ..errors import PipelineError
from ._common import RETRY_KW


def build_ssml(text: str, lang: str, voice: str, rate: float) -> str:
    n = round((rate - 1.0) * 100)
    rate_str = f"+{n}%" if n >= 0 else f"{n}%"
    return (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{XML_LANG[lang]}">'
        f'<voice name="{voice}"><prosody rate="{rate_str}">{escape(text)}</prosody></voice></speak>'
    )


class AzureTTS:
    name = "azure"

    def __init__(self) -> None:
        s = get_settings()
        self.url = f"https://{s.AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
        self.headers = {
            "Ocp-Apim-Subscription-Key": s.AZURE_SPEECH_KEY,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "riff-48khz-16bit-mono-pcm",
            "User-Agent": "11sport-dubber",
        }
        self.client = httpx.Client(timeout=60)

    def voice(self, lang: str, gender: str) -> str:
        return AZURE_VOICES[(lang, gender)]

    @retry(**RETRY_KW)
    def _post(self, ssml: str) -> bytes:
        r = self.client.post(self.url, headers=self.headers, content=ssml.encode("utf-8"))
        if r.status_code in (401, 403):
            raise PipelineError("Azure Speech từ chối truy cập: kiểm tra AZURE_SPEECH_KEY/AZURE_SPEECH_REGION")
        r.raise_for_status()
        return r.content

    def synthesize(self, text: str, lang: str, gender: str, rate: float, out_path: Path) -> None:
        audio = self._post(build_ssml(text, lang, self.voice(lang, gender), rate))
        if len(audio) < 100:
            raise PipelineError("Azure TTS trả về audio rỗng")
        out_path.write_bytes(audio)
