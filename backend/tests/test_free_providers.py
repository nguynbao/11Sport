from __future__ import annotations

from app.constants import normalize_lang
from app.providers.base import TranslateItem
from app.providers.free import EdgeFreeTTS, GoogleFreeTranslator, _apply_glossary


def test_normalize_lang():
    assert normalize_lang("vietnamese") == "vi"
    assert normalize_lang("VI") == "vi"
    assert normalize_lang("en-US") == "en"
    assert normalize_lang(None) == ""


def test_glossary_substitution_longest_first():
    g = [("11Sport", "11Sport"), ("giày chạy", "running shoes"), ("giày", "shoes")]
    assert _apply_glossary("Giày chạy của 11sport", g) == "running shoes của 11Sport"


def test_google_translate_batches_lines(monkeypatch):
    calls: list[str] = []

    def fake(self, text, source, target):
        calls.append(text)
        return "\n".join(f"T:{ln}" for ln in text.split("\n"))

    monkeypatch.setattr(GoogleFreeTranslator, "_translate", fake)
    items = [TranslateItem(i, f"câu {i}", 40) for i in range(5)]
    out = GoogleFreeTranslator().translate(items, "vietnamese", "en", [])
    assert out == {i: f"T:câu {i}" for i in range(5)}
    assert len(calls) == 1


def test_google_translate_falls_back_per_line(monkeypatch):
    def fake(self, text, source, target):
        return text.replace("\n", " ") + "!"  # line structure lost

    monkeypatch.setattr(GoogleFreeTranslator, "_translate", fake)
    items = [TranslateItem(i, f"a{i}", 40) for i in range(3)]
    out = GoogleFreeTranslator().translate(items, "en", "vi", [])
    assert out == {0: "a0!", 1: "a1!", 2: "a2!"}


def test_edge_tts_rate_and_conversion(monkeypatch, tmp_path):
    seen = {}

    def fake_save(self, text, voice, rate, mp3):
        seen.update(voice=voice, rate=rate)
        mp3.write_bytes(b"x" * 200)

    def fake_conv(src, out):
        out.write_bytes(src.read_bytes())

    monkeypatch.setattr(EdgeFreeTTS, "_save", fake_save)
    monkeypatch.setattr("app.media.ffmpeg.to_wav48_mono", fake_conv)
    out = tmp_path / "a.wav"
    EdgeFreeTTS().synthesize("Xin chào", "vi", "male", 1.2, out)
    assert seen == {"voice": "vi-VN-NamMinhNeural", "rate": "+20%"}
    assert out.exists()
    assert not list(tmp_path.glob("*.mp3"))
