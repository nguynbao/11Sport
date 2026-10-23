"""Timing-fit algorithm (plan 8.3). Pure logic: all I/O is injected via callbacks so it is unit-testable."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..text.budget import count_chars

MAX_TTS_RATE = 1.22
SPEEDUP_RATIO = 1.18
ATEMPO_EXTRA = 1.08
MAX_TOTAL_SPEED = 1.28
MIN_SLOW_RATE = 0.84
MAX_SHORTEN = 2
NEXT_GUARD = 0.05

# Exact-sync mode (used when the original speech end is known).
SYNC_TOL = 0.01          # voice-over length must be within ±1% of the original speech (99-100% sync)
SYNC_MIN_TTS_RATE = 0.60
SYNC_MAX_TTS_RATE = 1.45
SYNC_MIN_SPEED = 0.50    # slowest total speed (drawn-out / sung lines)
SYNC_MAX_SPEED = 1.95    # fastest total speed before truncation
SYNC_MAX_SPEED_SHORT = 2.00  # very short exclamations
SHORT_LINE_SEC = 1.2
SYNC_SHORTEN_AT = 1.18   # above this speed, prune fillers/commas first

# synth(text, rate) -> (audio_handle, duration)
SynthFn = Callable[[str, float], tuple[Any, float]]
# shorten(text, max_chars) -> new_text
ShortenFn = Callable[[str, int], str]
# atempo(audio_handle, factor) -> (audio_handle, duration)
AtempoFn = Callable[[Any, float], tuple[Any, float]]
# truncate(audio_handle, max_sec) -> (audio_handle, duration)
TruncateFn = Callable[[Any, float], tuple[Any, float]]


@dataclass
class FitResult:
    text: str
    audio: Any
    duration: float
    speed: float
    status: str  # ok | sped_up | shortened | overflow
    shortened: bool


def available_time(start: float, next_start: float) -> float:
    return max(0.3, (next_start - NEXT_GUARD) - start)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _match_duration(text: str, target: float, base_dur: float, base_audio: Any,
                    synth: SynthFn, atempo: AtempoFn,
                    max_speed: float = SYNC_MAX_SPEED) -> tuple[Any, float, float]:
    """Make `text` last `target` seconds (±SYNC_TOL) using coarse TTS rate + fine ffmpeg atempo.

    Returns (audio, duration, total_speed) where total_speed = base_dur / duration.
    """
    need = base_dur / target
    if abs(need - 1.0) <= SYNC_TOL:
        return base_audio, base_dur, 1.0
    need = _clamp(need, SYNC_MIN_SPEED, max_speed)
    rate = round(_clamp(need, SYNC_MIN_TTS_RATE, SYNC_MAX_TTS_RATE), 2)
    audio, d = (synth(text, rate) if abs(rate - 1.0) > 0.01 else (base_audio, base_dur))
    goal = base_dur / need
    f = d / goal
    if abs(f - 1.0) > SYNC_TOL / 2:
        audio, d = atempo(audio, f)
    return audio, d, base_dur / d if d > 0 else 1.0


def _fit_exact(text: str, target: float, available: float, lang: str, synth: SynthFn,
               shorten: ShortenFn, atempo: AtempoFn, truncate: TruncateFn) -> FitResult:
    shortened = False
    max_speed = SYNC_MAX_SPEED_SHORT if target < SHORT_LINE_SEC else SYNC_MAX_SPEED
    audio, d = synth(text, 1.0)
    for _ in range(MAX_SHORTEN):
        if d / target <= SYNC_SHORTEN_AT:
            break
        n = count_chars(text, lang)
        new_max = max(4, int(n * target / d * 0.95))
        new_text = shorten(text, new_max).strip()
        if not new_text or new_text == text:
            break
        text, shortened = new_text, True
        audio, d = synth(text, 1.0)

    genuinely_overflowed = (d / target) > max_speed * 1.05
    audio, dur, speed = _match_duration(text, target, d, audio, synth, atempo, max_speed)
    if dur > target + 0.015:
        audio, dur = truncate(audio, target)
        if genuinely_overflowed:
            return FitResult(text, audio, dur, round(speed, 3), "overflow", shortened)
    elif dur > available + 1e-6:
        audio, dur = truncate(audio, available)
    status = "shortened" if shortened else ("sped_up" if speed > 1.0 + SYNC_TOL * 2 else "ok")
    return FitResult(text, audio, dur, round(speed, 3), status, shortened)


def fit_segment(
    text: str,
    start: float,
    next_start: float,
    lang: str,
    synth: SynthFn,
    shorten: ShortenFn,
    atempo: AtempoFn,
    truncate: TruncateFn,
    orig_end: float | None = None,
) -> FitResult:
    available = available_time(start, next_start)
    if orig_end is not None:
        target = max(0.3, min(orig_end, next_start) - start)
        return _fit_exact(text, target, max(available, target), lang, synth, shorten, atempo, truncate)
    shortened = False

    def done(audio: Any, dur: float, speed: float, sped: bool) -> FitResult:
        status = "shortened" if shortened else ("sped_up" if sped else "ok")
        return FitResult(text, audio, dur, round(speed, 3), status, shortened)

    d = 0.0
    for attempt in range(MAX_SHORTEN + 1):
        audio, d = synth(text, 1.0)
        if d <= available:
            return done(audio, d, 1.0, False)
        ratio = d / available
        if ratio <= SPEEDUP_RATIO:
            rate = min(MAX_TTS_RATE, ratio + 0.01)
            audio2, d2 = synth(text, rate)
            if d2 <= available:
                return done(audio2, d2, ratio, True)
            f = d2 / available
            if f <= ATEMPO_EXTRA:
                audio3, d3 = atempo(audio2, f)
                if d3 > available:
                    audio3, d3 = truncate(audio3, available)
                return done(audio3, d3, rate * f, True)
        if attempt < MAX_SHORTEN:
            n = count_chars(text, lang)
            new_max = max(4, int(n * available / d * 0.92))
            new_text = shorten(text, new_max).strip()
            if new_text and new_text != text:
                text = new_text
                shortened = True
                continue
            # LLM could not shorten: stop trying
            break

    # Last resort: fastest TTS + atempo up to total MAX_TOTAL_SPEED, then cut with fade-out.
    audio, d = synth(text, MAX_TTS_RATE)
    speed = MAX_TTS_RATE
    if d > available:
        f = min(MAX_TOTAL_SPEED / MAX_TTS_RATE, d / available)
        if f > 1.001:
            audio, d = atempo(audio, f)
            speed *= f
    if d <= available:
        res = done(audio, d, speed, True)
        return res
    audio, d = truncate(audio, available)
    return FitResult(text, audio, d, round(speed, 3), "overflow", shortened)
