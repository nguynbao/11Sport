from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pysubs2

EN_LINE = 42
EN_LINE_PORTRAIT = 32
ZH_LINE = 16
ZH_LINE_PORTRAIT = 12
MAX_LINES = 2
ZH_BREAK_AFTER = "，。！？、；：,.!?;:"
SPLIT_PUNCT = (". ", "! ", "? ", "; ", ": ", ", ", "。", "！", "？", "；", "，", "、")
MIN_CUE = 0.8


@dataclass
class CueSource:
    start: float          # placed_start (or segment start)
    end: float            # original segment end
    placed_end: float     # end of voice-over
    text: str


@dataclass
class Cue:
    start: float
    end: float
    lines: list[str]


def wrap_zh(text: str, width: int = ZH_LINE) -> list[str]:
    text = "".join(text.split()) if all(ord(c) > 0x2E80 or not c.isalpha() for c in text) else text.strip()
    lines: list[str] = []
    rest = text
    while len(rest) > width:
        window = rest[:width]
        cut = max((i for i, ch in enumerate(window) if ch in ZH_BREAK_AFTER), default=-1)
        if cut < width // 3:
            cut = width - 1
        lines.append(rest[: cut + 1].strip())
        rest = rest[cut + 1 :].strip()
    if rest:
        lines.append(rest)
    return lines


def wrap_text(text: str, lang: str, max_width: int | None = None) -> list[str]:
    if lang == "zh":
        w = max_width or ZH_LINE
        return wrap_zh(text, w)
    w = max_width or EN_LINE
    s = text.strip()
    lines = textwrap.wrap(s, width=w, break_long_words=True) or [s]
    if len(lines) == 2:
        # Balance 2-line subtitles so we don't get one very long line and a 1-word orphan.
        target = max(w // 2, math.ceil(len(s) / 2) + 3)
        if target < w:
            balanced = textwrap.wrap(s, width=target, break_long_words=True)
            if len(balanced) == 2 and all(len(ln) <= w for ln in balanced):
                return balanced
    return lines


def _split_two(text: str, lang: str) -> tuple[str, str]:
    """Split text near the middle, preferring sentence/clause punctuation."""
    s = text.strip()
    mid = len(s) // 2
    lo, hi = max(1, int(len(s) * 0.28)), min(len(s) - 1, int(len(s) * 0.72))
    best_idx = None
    best_score = float("inf")
    for p in SPLIT_PUNCT:
        idx = s.find(p, lo, hi)
        while idx != -1:
            cut = idx + len(p.rstrip())
            score = abs(cut - mid) - (8 if p.strip() in ".!?。！？" else 0)
            if score < best_score:
                best_score = score
                best_idx = cut
            idx = s.find(p, idx + 1, hi)
    if best_idx is not None:
        return s[:best_idx].strip(), s[best_idx:].strip()
    if lang == "zh":
        return s[:mid].strip(), s[mid:].strip()
    spaces = [i for i, ch in enumerate(s) if ch == " "]
    if not spaces:
        return s[:mid], s[mid:]
    cut = min(spaces, key=lambda i: abs(i - mid))
    return s[:cut].strip(), s[cut + 1 :].strip()


def _cue_chunks(text: str, lang: str, line_width: int) -> list[list[str]]:
    lines = wrap_text(text, lang, line_width)
    if not lines:
        return []
    if len(lines) <= MAX_LINES:
        return [lines]
    if len(lines) <= MAX_LINES * 2:
        a, b = _split_two(text, lang)
        la, lb = wrap_text(a, lang, line_width), wrap_text(b, lang, line_width)
        if len(la) <= MAX_LINES and len(lb) <= MAX_LINES and la and lb:
            return [la, lb]
    return [lines[k : k + MAX_LINES] for k in range(0, len(lines), MAX_LINES)]


def build_cues(
    sources: list[CueSource],
    lang: str,
    video_duration: float | None = None,
    width: int = 1280,
    height: int = 720,
) -> list[Cue]:
    portrait = width < height
    line_width = (ZH_LINE_PORTRAIT if portrait else ZH_LINE) if lang == "zh" else (
        EN_LINE_PORTRAIT if portrait else EN_LINE
    )
    cues: list[Cue] = []
    for i, src in enumerate(sources):
        nxt = sources[i + 1].start if i + 1 < len(sources) else (video_duration or float("inf"))
        start = src.start
        end = max(src.end, src.placed_end)
        end = min(end, nxt - 0.02)
        end = max(end, start + MIN_CUE)
        if i + 1 < len(sources):
            end = min(end, max(start + 0.1, nxt - 0.02))
        if video_duration:
            end = min(end, video_duration)
        chunks = _cue_chunks(src.text, lang, line_width)
        if not chunks:
            continue
        if len(chunks) == 1:
            cues.append(Cue(start, end, chunks[0]))
            continue
        weights = [max(8, sum(len(x) for x in c)) for c in chunks]
        total = float(sum(weights)) or 1.0
        t = start
        for c, w in zip(chunks, weights, strict=True):
            d = (end - start) * w / total
            cues.append(Cue(t, t + d, c))
            t += d
    return cues


def _to_ssa(cues: list[Cue]) -> pysubs2.SSAFile:
    subs = pysubs2.SSAFile()
    for c in cues:
        subs.append(pysubs2.SSAEvent(start=round(c.start * 1000), end=round(c.end * 1000),
                                     text=r"\N".join(c.lines)))
    return subs


def write_srt(cues: list[Cue], path: Path) -> None:
    _to_ssa(cues).save(str(path), format_="srt")


# Noto Sans covers full Vietnamese diacritics; the CJK font is only needed for Chinese.
FONT_BY_LANG = {"zh": "Noto Sans CJK SC"}
DEFAULT_FONT = "Noto Sans"


def write_ass(cues: list[Cue], path: Path, width: int, height: int, lang: str = "en") -> None:
    subs = _to_ssa(cues)
    subs.info["PlayResX"] = str(width)
    subs.info["PlayResY"] = str(height)
    subs.info["WrapStyle"] = "0"  # smart wrapping inside [MarginL, PlayResX - MarginR] so text never clips
    short_side = min(width, height)
    fontsize = max(18, round(short_side * 0.052))
    margin_h = max(16, round(width * 0.06))
    style = pysubs2.SSAStyle(
        fontname=FONT_BY_LANG.get(lang, DEFAULT_FONT),
        fontsize=fontsize,
        primarycolor=pysubs2.Color(255, 255, 255, 0),
        outlinecolor=pysubs2.Color(0, 0, 0, 0),
        backcolor=pysubs2.Color(0, 0, 0, 128),
        outline=max(1, round(fontsize / 14)),
        shadow=0,
        alignment=pysubs2.Alignment.BOTTOM_CENTER,
        marginl=margin_h,
        marginr=margin_h,
        marginv=round(height * 0.06),
        borderstyle=1,
    )
    subs.styles["Default"] = style
    subs.save(str(path), format_="ass")

