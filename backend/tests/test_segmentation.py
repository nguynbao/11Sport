import pytest

from app.errors import PipelineError
from app.providers.base import ASRResult, ASRSegment, ASRWord
from app.text.segmentation import build_segments


def test_filters_hallucination_and_empty():
    r = ASRResult("vietnamese", [
        ASRSegment(0, 2, "Xin chào.", -0.2, 0.1),
        ASRSegment(2.5, 4, "Cảm ơn đã xem", -1.5, 0.9),   # hallucination
        ASRSegment(4.5, 5, "   ", -0.1, 0.0),
        ASRSegment(6, 6.1, "a", -0.1, 0.0),               # too short
    ])
    segs = build_segments(r)
    assert [s.text for s in segs] == ["Xin chào."]


def test_merge_fragments_small_gap():
    r = ASRResult("vietnamese", [
        ASRSegment(0, 2, "Hôm nay chúng ta"),
        ASRSegment(2.2, 4, "sẽ nói về giày."),
    ])
    segs = build_segments(r)
    assert len(segs) == 1
    assert segs[0].text == "Hôm nay chúng ta sẽ nói về giày."
    assert (segs[0].start, segs[0].end) == (0, 4)


def test_no_merge_large_gap():
    r = ASRResult("vietnamese", [ASRSegment(0, 2, "Hôm nay chúng ta"), ASRSegment(2.5, 4, "sẽ nói")])
    assert len(build_segments(r)) == 2


def test_no_merge_after_sentence_end():
    r = ASRResult("vietnamese", [ASRSegment(0, 2, "Xin chào."), ASRSegment(2.1, 4, "Tôi là An")])
    assert len(build_segments(r)) == 2


def test_split_long_segment_with_words():
    words = [ASRWord(i * 1.0, i * 1.0 + 0.8, f"w{i}") for i in range(20)]
    words[9] = ASRWord(9.0, 9.3, "w9")  # bigger gap after w9 (9.3 -> 10.0)
    r = ASRResult("vietnamese", [ASRSegment(0, 19.8, " ".join(w.word for w in words))], words)
    segs = build_segments(r)
    assert len(segs) >= 2
    assert all(s.end - s.start <= 12.0 for s in segs)
    assert " ".join(s.text for s in segs) == " ".join(w.word for w in words)


def test_nothing_left_raises():
    with pytest.raises(PipelineError):
        build_segments(ASRResult("vietnamese", [ASRSegment(0, 2, "", 0, 0)]))
