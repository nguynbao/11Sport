"""Free demo providers — no API key required.

- ASR: faster-whisper running locally on CPU (model downloaded once from Hugging Face).
- Translation: Google Translate public "gtx" endpoint (needs internet, no key).
- TTS: Microsoft Edge read-aloud neural voices via edge-tts (needs internet, no key).

Quality/limits are fine for demos and short (1-3 min) videos, but these free endpoints have no SLA
and may rate-limit; switch to PROVIDER_MODE=real for production use.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
from pathlib import Path

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import get_settings
from ..constants import AZURE_VOICES, normalize_lang
from ..errors import PipelineError
from ..media import ffmpeg as F
from .base import ASRResult, ASRSegment, ASRWord, Glossary, TranslateItem

log = logging.getLogger(__name__)

_FREE_RETRY = {
    "retry": retry_if_exception_type(Exception),
    "stop": stop_after_attempt(4),
    "wait": wait_exponential(multiplier=2, min=2, max=20),
    "reraise": True,
}


# ------------------------------------------------------------------ ASR
_model_lock = threading.Lock()
_model = None


def _whisper_model():
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel

            s = get_settings()
            log.info("loading faster-whisper model %r (first run downloads it)", s.WHISPER_MODEL)
            _model = WhisperModel(s.WHISPER_MODEL, device="cpu", compute_type=s.WHISPER_COMPUTE_TYPE)
        return _model


def _decode_16k_mono(path: Path):
    """Decode to float32 16 kHz mono with our ffmpeg (avoids faster-whisper's PyAV dependency quirks)."""
    import subprocess

    import numpy as np

    proc = subprocess.run(
        [F.ffmpeg_bin(), "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000",
         "-f", "f32le", "-"],
        capture_output=True, check=False,
    )
    if proc.returncode != 0:
        raise PipelineError(f"Không giải mã được audio: {proc.stderr.decode(errors='ignore')[-500:]}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


class LocalWhisperASR:
    def transcribe(self, audio_path: Path, prompt: str | None) -> ASRResult:
        model = _whisper_model()
        seg_iter, info = model.transcribe(
            _decode_16k_mono(audio_path),
            word_timestamps=True,
            vad_filter=True,
            initial_prompt=prompt or None,
            beam_size=5,
        )
        segments: list[ASRSegment] = []
        words: list[ASRWord] = []
        for s in seg_iter:  # generator: transcription happens while iterating
            segments.append(ASRSegment(float(s.start), float(s.end), s.text or "",
                                       float(s.avg_logprob or 0.0), float(s.no_speech_prob or 0.0)))
            for w in s.words or []:
                words.append(ASRWord(float(w.start), float(w.end), w.word or ""))
        return ASRResult(language=info.language or "", segments=segments, words=words)


# ------------------------------------------------------------------ translation
_SEP = "\n"


def _apply_glossary(text: str, glossary: Glossary) -> str:
    # No LLM to obey instructions, so substitute glossary terms before translating.
    for src, tgt in sorted(glossary, key=lambda x: -len(x[0])):
        text = re.sub(re.escape(src), tgt, text, flags=re.IGNORECASE)
    return text


class GoogleFreeTranslator:
    URL = "https://translate.googleapis.com/translate_a/single"

    @retry(**_FREE_RETRY)
    def _translate(self, text: str, source: str, target: str) -> str:
        import httpx

        r = httpx.post(self.URL, params={"client": "gtx", "sl": source, "tl": target, "dt": "t"},
                       data={"q": text}, timeout=30,
                       headers={"User-Agent": "Mozilla/5.0 (11sport-dubber demo)"})
        r.raise_for_status()
        data = r.json()
        return "".join(chunk[0] for chunk in (data[0] or []) if chunk and chunk[0])

    def translate(self, items: list[TranslateItem], source_lang: str, target_lang: str,
                  glossary: Glossary) -> dict[int, str]:
        if not items:
            return {}
        src = normalize_lang(source_lang) or "auto"
        if src not in ("en", "vi", "zh", "ja", "ko", "fr", "de", "es", "th", "id"):
            src = "auto"
        if src == "zh":
            src = "zh-CN"
        texts = [_apply_glossary(it.text.replace(_SEP, " ").strip(), glossary) for it in items]
        out: dict[int, str] = {}
        # Translate in chunks of lines (keeps sentence context, few requests, < 5000-char limit).
        chunk: list[int] = []
        size = 0

        def flush() -> None:
            nonlocal chunk, size
            if not chunk:
                return
            joined = _SEP.join(texts[i] for i in chunk)
            try:
                lines = [ln.strip() for ln in self._translate(joined, src, target_lang).split(_SEP)]
            except Exception as e:
                raise PipelineError(f"Dịch miễn phí (Google) lỗi: {e}") from e
            if len(lines) != len(chunk):
                # Line structure was not preserved: fall back to one request per line.
                lines = [self._translate(texts[i], src, target_lang).strip() for i in chunk]
            for i, line in zip(chunk, lines, strict=True):
                out[items[i].id] = line or texts[i]
            chunk, size = [], 0

        for i, t in enumerate(texts):
            if size + len(t) > 3500:
                flush()
            chunk.append(i)
            size += len(t) + 1
        flush()
        return out

    def shorten(self, text: str, target_lang: str, max_chars: int, glossary: Glossary) -> str:
        # Free mode has no LLM to rewrite; the fitter will speed up the voice instead.
        return text


_FREE_LLM_TARGET = {
    "en": "natural spoken American English",
    "vi": "natural spoken Vietnamese (standard, with full diacritics)",
    "zh": "Simplified Chinese (Mandarin)",
}

_FREE_LLM_SYS = """You are an expert subtitle and voice-over dubbing translator.
Translate each numbered item from {src} into {tgt}.
Rules:
1. Context-aware: read all lines together to understand the video's topic (sports, coaching, tutorials, comedy, slang) and fix obvious ASR transcription typos/homophones.
2. Voice-over timing: each item has `max_chars`. Keep the translation concise and natural so a voice actor can speak it comfortably within `max_chars` characters at 1.0x speed.
3. Preserve exact item IDs. Return a JSON object mapping each string ID to its translated text, e.g. {{"0": "...", "1": "..."}}.
4. Glossary (MUST follow): {glossary}"""


class FreeLLMTranslator:
    """Free 20B Reasoning LLM translator (GPT-OSS 20B via Pollinations OpenAI API) with Google NMT fallback."""

    LLM_URL = "https://text.pollinations.ai/openai"
    LLM_MODEL = "openai-fast"
    BATCH_SIZE = 80
    MIN_INTERVAL_SEC = 16.0
    _last_call_ts: float = 0.0

    def __init__(self) -> None:
        self._fallback = GoogleFreeTranslator()

    @staticmethod
    def _glossary_str(glossary: Glossary) -> str:
        pairs = [f'"{s}" => "{t}"' for s, t in glossary if s and t]
        return "; ".join(pairs) if pairs else "none"

    def _call_llm_json(self, system: str, user_obj: dict) -> dict:
        import json
        import time

        import httpx

        elapsed = time.monotonic() - FreeLLMTranslator._last_call_ts
        if elapsed < self.MIN_INTERVAL_SEC and FreeLLMTranslator._last_call_ts > 0:
            time.sleep(self.MIN_INTERVAL_SEC - elapsed)

        payload = {
            "model": self.LLM_MODEL,
            "temperature": 0.2,
            "reasoning_effort": "low",
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)},
            ],
        }
        r = httpx.post(self.LLM_URL, json=payload, timeout=45.0)
        FreeLLMTranslator._last_call_ts = time.monotonic()
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        content = msg.get("content") or ""
        if not content:
            raise ValueError("LLM returned empty content")
        return json.loads(content)

    def _translate_batch_llm(
        self, items: list[TranslateItem], source_lang: str, target_lang: str, glossary: Glossary
    ) -> dict[int, str]:
        system = _FREE_LLM_SYS.format(
            src=source_lang or "the source language",
            tgt=_FREE_LLM_TARGET.get(target_lang, target_lang),
            glossary=self._glossary_str(glossary),
        )
        user_obj = {
            str(it.id): {"text": _apply_glossary(it.text.strip(), glossary), "max_chars": it.max_chars}
            for it in items
        }
        data = self._call_llm_json(system, user_obj)
        out: dict[int, str] = {}
        if isinstance(data.get("items"), list):
            for entry in data["items"]:
                if isinstance(entry, dict) and "id" in entry and "text" in entry:
                    out[int(entry["id"])] = str(entry["text"]).strip()
        else:
            for k, v in data.items():
                if str(k).isdigit():
                    if isinstance(v, str) and v.strip():
                        out[int(k)] = v.strip()
                    elif isinstance(v, dict) and isinstance(v.get("text"), str) and v["text"].strip():
                        out[int(k)] = v["text"].strip()
        return out

    def translate(
        self, items: list[TranslateItem], source_lang: str, target_lang: str, glossary: Glossary
    ) -> dict[int, str]:
        if not items:
            return {}
        out: dict[int, str] = {}
        for i in range(0, len(items), self.BATCH_SIZE):
            batch = items[i : i + self.BATCH_SIZE]
            try:
                got = self._translate_batch_llm(batch, source_lang, target_lang, glossary)
                out.update(got)
            except Exception as e:  # noqa: BLE001
                log.warning("Free 20B LLM batch translation fallback (%s)", e)
            missing = [it for it in batch if it.id not in out or not out[it.id]]
            if missing:
                out.update(self._fallback.translate(missing, source_lang, target_lang, glossary))
        return out

    def shorten(self, text: str, target_lang: str, max_chars: int, glossary: Glossary) -> str:
        """Fast local spoken-filler and pause-comma pruning so synthesis fits tight slots without LLM rate limits."""
        s = text.strip()
        if target_lang == "vi":
            fillers = (
                " nhé các bạn", " nha các bạn", " các bạn nhé", " các bạn nha",
                " cho mình nhé", " cho mình nha", " luôn nhé", " luôn nha",
                " nhé", " nha", " ạ", " đấy", " đó", " này", " thì", " mà",
                " của mình", " cho các bạn", " các bạn",
            )
        else:
            fillers = (
                " ladies and gentlemen", " everyone", " you guys", " right here",
                " right now", " for yourself", " for me", " actually", " basically",
                " really", " very", " just", " please", " go ahead and",
            )
        for f in fillers:
            if len(s) <= max_chars:
                break
            s = re.sub(re.escape(f), "", s, flags=re.IGNORECASE).strip()
            s = re.sub(r"\s{2,}", " ", s)
        # Strip internal commas/semicolons/dashes that cause TTS to insert 350-450ms pauses per comma
        if "," in s or ";" in s or " - " in s:
            s = re.sub(r"[,;]+\s*", " ", s)
            s = re.sub(r"\s+-\s+", " ", s)
            s = re.sub(r"\s{2,}", " ", s).strip()
        return s or text


# ------------------------------------------------------------------ TTS
class EdgeFreeTTS:
    name = "edge"

    def voice(self, lang: str, gender: str) -> str:
        base = AZURE_VOICES.get((lang, gender)) or AZURE_VOICES[(lang, "male")]
        return f"{base}-adam-v2" if gender == "adam" else f"{base}-v2"

    @staticmethod
    def _google_vi_female(text: str, rate: str, mp3: Path) -> bool:
        """Native Vietnamese female TTS fallback when Microsoft's vi-VN-HoaiMyNeural is unavailable."""
        import subprocess

        import httpx

        try:
            r = httpx.get(
                "https://translate.google.com/translate_tts",
                params={"ie": "UTF-8", "tl": "vi", "client": "tw-ob", "q": text[:200]},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15.0,
            )
            if r.status_code != 200 or len(r.content) < 100:
                return False
            pct = int(rate.replace("%", "").replace("+", "") or "0")
            factor = max(0.5, min(2.0, 1.0 + pct / 100.0))
            if abs(factor - 1.0) <= 0.01:
                mp3.write_bytes(r.content)
                return True
            raw = mp3.with_suffix(".graw.mp3")
            raw.write_bytes(r.content)
            try:
                proc = subprocess.run(
                    [F.ffmpeg_bin(), "-nostdin", "-y", "-v", "error", "-i", str(raw),
                     "-filter:a", f"atempo={factor:.4f}", "-c:a", "libmp3lame", "-q:a", "2", str(mp3)],
                    capture_output=True, check=False,
                )
                return proc.returncode == 0 and mp3.exists() and mp3.stat().st_size > 100
            finally:
                raw.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            return False

    @retry(**_FREE_RETRY)
    def _save(self, text: str, voice: str, rate: str, mp3: Path, pitch: str = "+0Hz") -> None:
        import edge_tts

        async def run() -> None:
            try:
                await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(str(mp3))
            except edge_tts.exceptions.NoAudioReceived:
                if "HoaiMy" in voice and self._google_vi_female(text, rate, mp3):
                    return
                # Multilingual neural voices (never monolingual JennyNeural which reads VI with English accent)
                fb_voice = "en-US-EmmaMultilingualNeural" if "HoaiMy" in voice or "Jenny" in voice else "en-US-AndrewMultilingualNeural"
                await edge_tts.Communicate(text, fb_voice, rate=rate, pitch=pitch).save(str(mp3))

        asyncio.run(run())
        if not mp3.exists() or mp3.stat().st_size < 100:
            raise RuntimeError("Edge TTS trả về audio rỗng")

    def synthesize(self, text: str, lang: str, gender: str, rate: float, out_path: Path) -> None:
        n = round((rate - 1.0) * 100)
        rate_str = f"+{n}%" if n >= 0 else f"{n}%"
        raw_voice = AZURE_VOICES.get((lang, gender)) or AZURE_VOICES[(lang, "male")]
        mp3 = out_path.with_suffix(".edge.mp3")
        try:
            if gender == "adam":
                self._save(text, raw_voice, rate_str, mp3, "-5Hz")
            else:
                self._save(text, raw_voice, rate_str, mp3)
        except Exception as e:
            raise PipelineError(f"Giọng đọc miễn phí (Edge TTS) lỗi: {e}") from e
        try:
            if gender == "adam":
                F.to_wav48_mono(mp3, out_path, adam_master=True)
            else:
                F.to_wav48_mono(mp3, out_path)
        finally:
            mp3.unlink(missing_ok=True)



