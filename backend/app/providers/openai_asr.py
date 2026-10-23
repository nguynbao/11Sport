from __future__ import annotations

from pathlib import Path

from tenacity import retry

from ..config import get_settings
from ._common import RETRY_KW, g, make_client
from .base import ASRResult, ASRSegment, ASRWord


class OpenAIASR:
    def __init__(self) -> None:
        self.client = make_client()

    @retry(**RETRY_KW)
    def transcribe(self, audio_path: Path, prompt: str | None) -> ASRResult:
        with open(audio_path, "rb") as f:
            kwargs = {
                "model": get_settings().OPENAI_ASR_MODEL,
                "file": f,
                "response_format": "verbose_json",
                "timestamp_granularities": ["segment", "word"],
            }
            if prompt:
                kwargs["prompt"] = prompt
            resp = self.client.audio.transcriptions.create(**kwargs)
        segments = [
            ASRSegment(
                start=float(g(s, "start", 0.0)),
                end=float(g(s, "end", 0.0)),
                text=str(g(s, "text", "") or ""),
                avg_logprob=float(g(s, "avg_logprob", 0.0) or 0.0),
                no_speech_prob=float(g(s, "no_speech_prob", 0.0) or 0.0),
            )
            for s in (g(resp, "segments") or [])
        ]
        words = [
            ASRWord(float(g(w, "start", 0.0)), float(g(w, "end", 0.0)), str(g(w, "word", "") or ""))
            for w in (g(resp, "words") or [])
        ]
        return ASRResult(language=str(g(resp, "language", "") or ""), segments=segments, words=words)
