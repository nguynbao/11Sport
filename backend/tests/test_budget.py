from app.text.budget import count_chars, max_chars, slot_seconds


def test_count_chars_zh_ignores_punct_and_space():
    assert count_chars("你好，世界！", "zh") == 4
    assert count_chars(" 你 好 ", "zh") == 2


def test_count_chars_en():
    assert count_chars("  Hello world ", "en") == 11


def test_max_chars():
    assert max_chars(2.0, "en") == 28
    assert max_chars(0.1, "zh") == 4
    assert max_chars(4.0, "zh") == 17


def test_slot_seconds_caps_trailing_gap():
    assert slot_seconds(0.0, 2.0, 5.0) == 2.6
    assert slot_seconds(0.0, 2.0, 2.3) == 2.3
