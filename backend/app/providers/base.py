from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class ASRWord:
    start: float
    end: float
    word: str


@dataclass
class ASRSegment:
    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0


@dataclass
class ASRResult:
    language: str
    segments: list[ASRSegment] = field(default_factory=list)
    words: list[ASRWord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "segments": [s.__dict__ for s in self.segments],
            "words": [w.__dict__ for w in self.words],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ASRResult:
        return cls(
            language=d.get("language") or "",
            segments=[ASRSegment(**s) for s in d.get("segments", [])],
            words=[ASRWord(**w) for w in d.get("words", [])],
        )


class ASRProvider(Protocol):
    def transcribe(self, audio_path: Path, prompt: str | None) -> ASRResult: ...


@dataclass
class TranslateItem:
    id: int
    text: str
    max_chars: int


Glossary = list[tuple[str, str]]


class Translator(Protocol):
    def translate(
        self, items: list[TranslateItem], source_lang: str, target_lang: str, glossary: Glossary
    ) -> dict[int, str]: ...

    def shorten(self, text: str, target_lang: str, max_chars: int, glossary: Glossary) -> str: ...


class TTSProvider(Protocol):
    name: str

    def synthesize(self, text: str, lang: str, gender: str, rate: float, out_path: Path) -> None:
        """Write WAV 48kHz mono pcm_s16le. rate=1.0 is normal speed; 1.0-1.3 allowed."""
        ...
