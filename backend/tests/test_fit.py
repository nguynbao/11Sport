from app.pipeline.fit import fit_segment

CPS = 15.0  # fake TTS: duration = len(text) / CPS / rate


def make_env(shorten_map=None):
    calls = {"synth": [], "shorten": [], "atempo": [], "truncate": []}

    def synth(text, rate):
        calls["synth"].append((text, rate))
        return ("audio", text, rate), len(text) / CPS / rate

    def shorten(text, mc):
        calls["shorten"].append((text, mc))
        if shorten_map is not None:
            return shorten_map(text, mc)
        return text[:mc]

    def atempo(audio, f):
        calls["atempo"].append(f)
        _, text, rate = audio
        return ("atempo", text, rate * f), len(text) / CPS / rate / f

    def truncate(audio, max_sec):
        calls["truncate"].append(max_sec)
        return ("cut",), max_sec

    return calls, synth, shorten, atempo, truncate


def run(text, available, shorten_map=None):
    calls, *fns = make_env(shorten_map)
    # available = next_start - 0.05 - start
    res = fit_segment(text, 0.0, available + 0.05, "en", *fns)
    return res, calls


def test_fits_directly():
    res, calls = run("x" * 30, 3.0)  # 2.0s
    assert res.status == "ok" and res.speed == 1.0
    assert len(calls["synth"]) == 1


def test_speed_up_small_ratio():
    res, calls = run("x" * 30, 2.0 / 1.15)  # ratio 1.15
    assert res.status == "sped_up"
    assert res.duration <= 2.0 / 1.15 + 1e-9
    assert calls["synth"][1][1] > 1.0
    assert not calls["shorten"]


def test_shorten_when_ratio_large():
    res, calls = run("x" * 48, 2.0)  # 3.2s, ratio 1.6
    assert calls["shorten"]
    assert res.status == "shortened"
    assert res.duration <= 2.0 + 1e-9
    assert res.shortened and len(res.text) < 48


def test_overflow_when_cannot_shorten():
    res, calls = run("x" * 90, 2.0, shorten_map=lambda t, mc: t)  # LLM returns same text
    assert res.status == "overflow"
    assert res.duration <= 2.0 + 1e-9
    assert calls["truncate"]


def run_exact(text, orig_dur, gap_after=1.0, shorten_map=None):
    calls, *fns = make_env(shorten_map)
    res = fit_segment(text, 10.0, 10.0 + orig_dur + gap_after, "en", *fns, orig_end=10.0 + orig_dur)
    return res, calls


def test_exact_sync_stretches_short_translation():
    res, _ = run_exact("x" * 30, 2.3)  # TTS 2.0s vs original 2.3s
    assert abs(res.duration - 2.3) / 2.3 <= 0.02


def test_exact_sync_compresses_long_translation():
    res, calls = run_exact("x" * 30, 1.7)  # TTS 2.0s vs original 1.7s
    assert abs(res.duration - 1.7) / 1.7 <= 0.02
    assert not calls["shorten"] and res.status == "sped_up"


def test_exact_sync_shortens_then_matches():
    res, calls = run_exact("x" * 60, 2.0)  # TTS 4.0s vs 2.0s -> needs shortening
    assert calls["shorten"] and res.shortened
    assert abs(res.duration - 2.0) / 2.0 <= 0.02


def test_exact_sync_already_matching_is_untouched():
    res, calls = run_exact("x" * 30, 2.01)
    assert res.speed == 1.0 and len(calls["synth"]) == 1 and not calls["atempo"]
