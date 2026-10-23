from __future__ import annotations

from pathlib import Path

from ..media import audio as A
from ..media.ffmpeg import audio_duration
from ..text.budget import CHAR_RATE
from .base import ASRResult, ASRSegment, ASRWord, Glossary, TranslateItem


class FakeASR:
    def transcribe(self, audio_path: Path, prompt: str | None) -> ASRResult:
        dur = audio_duration(audio_path)
        segs: list[ASRSegment] = []
        words: list[ASRWord] = []
        t, i = 0.0, 0
        while t + 1.0 < dur:
            end = min(dur, t + 4.0) - 0.3
            text = f"Đây là câu thử nghiệm số {i}."
            segs.append(ASRSegment(t, end, text, -0.2, 0.01))
            toks = text.split()
            step = (end - t) / len(toks)
            for k, w in enumerate(toks):
                words.append(ASRWord(t + k * step, t + (k + 1) * step, w))
            t += 4.0
            i += 1
        return ASRResult("vietnamese", segs, words)


class FakeTranslator:
    def translate(self, items: list[TranslateItem], source_lang: str, target_lang: str,
                  glossary: Glossary) -> dict[int, str]:
        out = {}
        for it in items:
            text = f"[{target_lang}] Test sentence {it.id}" if target_lang == "en" else f"Câu thử nghiệm số {it.id}"
            out[it.id] = text[: it.max_chars]
        return out

    def shorten(self, text: str, target_lang: str, max_chars: int, glossary: Glossary) -> str:
        return text[:max_chars]


class FakeTTS:
    name = "fake"

    def synthesize(self, text: str, lang: str, gender: str, rate: float, out_path: Path) -> None:
        dur = max(0.2, len(text) / CHAR_RATE.get(lang, 15.0) / max(rate, 0.1))
        A.write_wav(out_path, A.sine(dur))
