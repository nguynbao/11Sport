from __future__ import annotations

import unicodedata

CHAR_RATE = {"en": 15.0, "vi": 14.0, "zh": 4.5}  # characters/second at natural speaking speed
MAX_TRAILING_GAP = 0.6


def count_chars(text: str, lang: str) -> int:
    if lang == "zh":
        return sum(1 for ch in text if not unicodedata.category(ch).startswith(("P", "Z")) and not ch.isspace())
    return len(text.strip())


def max_chars(slot_sec: float, lang: str) -> int:
    return max(4, int(slot_sec * CHAR_RATE[lang] * 0.95))


def slot_seconds(start: float, end: float, next_start: float) -> float:
    """Segment slot = own duration + up to 0.6s of trailing silence (never past next segment)."""
    return max(0.0, min(next_start, end + MAX_TRAILING_GAP) - start)
