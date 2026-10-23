import itertools

import pysubs2

from app.media.subtitles import CueSource, build_cues, wrap_text, write_ass, write_srt


def test_wrap_en_42():
    text = ("This is a fairly long English subtitle line that definitely needs to be wrapped "
            "into multiple lines ok")
    assert len(text) >= 100
    lines = wrap_text(text, "en")
    assert all(len(line) <= 42 for line in lines)


def test_wrap_zh_16():
    text = "这是一个很长的中文字幕句子，用来测试自动换行功能是否正常工作，我们希望每行不超过十六个字。"
    lines = wrap_text(text, "zh")
    assert all(len(line) <= 16 for line in lines)
    assert "".join(lines) == text


def test_cues_max_two_lines_and_no_overlap():
    long_en = "word " * 40
    sources = [CueSource(0.0, 4.0, 4.5, long_en.strip()), CueSource(5.0, 7.0, 6.5, "Short one.")]
    cues = build_cues(sources, "en", 10.0)
    assert all(len(c.lines) <= 2 for c in cues)
    for a, b in itertools.pairwise(cues):
        assert a.end <= b.start + 1e-6
    assert cues[-1].end <= 10.0


def test_cue_min_duration_and_clamp_next():
    sources = [CueSource(0.0, 0.2, 0.3, "Hi"), CueSource(0.5, 2.0, 2.0, "There")]
    cues = build_cues(sources, "en", 3.0)
    assert cues[0].end <= 0.5 - 0.02 + 1e-6


def test_srt_and_ass_roundtrip(tmp_path):
    cues = build_cues([CueSource(0.0, 2.0, 2.0, "你好，世界"), CueSource(2.5, 4.0, 4.0, "再见")], "zh", 5.0)
    write_srt(cues, tmp_path / "a.srt")
    write_ass(cues, tmp_path / "a.ass", 1280, 720, "zh")
    srt = pysubs2.load(str(tmp_path / "a.srt"))
    assert len(srt) == 2 and srt[0].plaintext == "你好，世界"
    ass = pysubs2.load(str(tmp_path / "a.ass"))
    assert ass.styles["Default"].fontname == "Noto Sans CJK SC"
    assert ass.info["PlayResY"] == "720"


def test_portrait_ass_font_and_margins_and_balanced_cues(tmp_path):
    text = "For those of you who are just learning to play, please take guest number 2 for yourself."
    cues = build_cues([CueSource(0.0, 4.0, 4.0, text)], "en", 10.0, width=576, height=1024)
    assert len(cues) == 2
    assert all(len(c.lines) <= 2 for c in cues)
    assert all(all(len(ln) <= 32 for ln in c.lines) for c in cues)
    # Neither cue should be a tiny 1-word orphan
    assert sum(len(ln) for ln in cues[1].lines) >= 20
    write_ass(cues, tmp_path / "p.ass", 576, 1024, "en")
    ass = pysubs2.load(str(tmp_path / "p.ass"))
    st = ass.styles["Default"]
    assert st.fontsize == round(576 * 0.052)
    assert st.marginl >= 30 and st.marginr >= 30
    assert ass.info["WrapStyle"] == "0"

