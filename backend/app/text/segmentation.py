from __future__ import annotations

import re
from dataclasses import dataclass

from ..errors import PipelineError
from ..providers.base import ASRResult, ASRWord

SENTENCE_END = (".", "?", "!", "。", "？", "！", "…")
MERGE_MAX_GAP = 0.4
MERGE_MAX_DUR = 10.0
SPLIT_MAX_DUR = 12.0
PAUSE_SPLIT_GAP = 0.45


@dataclass
class Seg:
    start: float
    end: float
    text: str


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _filter(result: ASRResult) -> list[Seg]:
    out: list[Seg] = []
    for s in result.segments:
        text = _norm(s.text)
        if not text:
            continue
        if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
            continue
        if s.end - s.start < 0.2:
            continue
        out.append(Seg(s.start, s.end, text))
    return out


def _merge(segs: list[Seg]) -> list[Seg]:
    if not segs:
        return []
    out = [Seg(segs[0].start, segs[0].end, segs[0].text)]
    for nxt in segs[1:]:
        cur = out[-1]
        if (
            not cur.text.endswith(SENTENCE_END)
            and nxt.start - cur.end < MERGE_MAX_GAP
            and nxt.end - cur.start <= MERGE_MAX_DUR
        ):
            cur.end = nxt.end
            cur.text = f"{cur.text} {nxt.text}"
        else:
            out.append(Seg(nxt.start, nxt.end, nxt.text))
    return out


def _words_text(words: list[ASRWord]) -> str:
    return _norm(" ".join(w.word.strip() for w in words))


def _split_pauses(seg: Seg, words: list[ASRWord]) -> list[Seg]:
    """Split a segment whenever the speaker pauses >= PAUSE_SPLIT_GAP seconds between words."""
    inside = [w for w in words if w.start >= seg.start - 0.05 and w.end <= seg.end + 0.05 and w.word.strip()]
    if len(inside) < 2:
        return [seg]
    parts: list[Seg] = []
    chunk: list[ASRWord] = [inside[0]]
    for w in inside[1:]:
        if w.start - chunk[-1].end >= PAUSE_SPLIT_GAP:
            txt = _words_text(chunk)
            if txt:
                s0 = chunk[0].start
                e0 = min(w.start - 0.05, max(chunk[-1].end, s0 + 0.4))
                parts.append(Seg(s0, e0, txt))
            chunk = [w]
        else:
            chunk.append(w)
    if chunk:
        txt = _words_text(chunk)
        if txt:
            s0 = chunk[0].start
            e0 = max(chunk[-1].end, seg.end, s0 + 0.4)
            parts.append(Seg(s0, e0, txt))
    return parts if len(parts) > 1 else [seg]


def _split_one(seg: Seg, words: list[ASRWord]) -> list[Seg]:
    if seg.end - seg.start <= SPLIT_MAX_DUR:
        return [seg]
    inside = [w for w in words if w.start >= seg.start - 0.05 and w.end <= seg.end + 0.05]
    if len(inside) < 2:
        return [seg]
    dur = seg.end - seg.start
    lo, hi = seg.start + 0.3 * dur, seg.start + 0.7 * dur
    best_i, best_gap = None, -1.0
    for i in range(len(inside) - 1):
        boundary = inside[i].end
        if lo <= boundary <= hi:
            gap = inside[i + 1].start - inside[i].end
            if gap > best_gap:
                best_i, best_gap = i, gap
    if best_i is None:
        # No boundary inside the 30-70% window: take the boundary closest to the middle.
        mid = seg.start + dur / 2
        best_i = min(range(len(inside) - 1), key=lambda i: abs(inside[i].end - mid))
    left_w, right_w = inside[: best_i + 1], inside[best_i + 1 :]
    left = Seg(seg.start, left_w[-1].end, _words_text(left_w))
    right = Seg(right_w[0].start, seg.end, _words_text(right_w))
    if left.end - left.start >= dur or right.end - right.start >= dur:
        return [seg]
    return _split_one(left, words) + _split_one(right, words)


MIN_SEG_DUR = 0.3


def _snap_to_words(segs: list[Seg], words: list[ASRWord]) -> list[Seg]:
    """Use word-level timestamps for precise speech boundaries and repair degenerate segments."""
    out: list[Seg] = []
    for i, s in enumerate(segs):
        inside = [w for w in words if w.word.strip() and w.start >= s.start - 0.05 and w.end <= s.end + 0.05]
        start, end = s.start, s.end
        if inside:
            start, end = inside[0].start, inside[-1].end
        if end - start < MIN_SEG_DUR:
            limit = segs[i + 1].start - 0.05 if i + 1 < len(segs) else end + MIN_SEG_DUR
            end = max(end, min(start + MIN_SEG_DUR, limit))
        out.append(Seg(start, max(end, start + 0.05), s.text))
    return out


def build_segments(result: ASRResult) -> list[Seg]:
    segs = _filter(result)
    if not segs:
        raise PipelineError("Không nhận diện được giọng nói trong video")
    segs = _merge(segs)
    out: list[Seg] = []
    for s in segs:
        for p in _split_pauses(s, result.words):
            out.extend(_split_one(p, result.words))
    out = [s for s in out if _norm(s.text)]
    if result.words:
        out = _snap_to_words(out, result.words)
    return [Seg(round(s.start, 3), round(s.end, 3), _norm(s.text)) for s in out]
