import numpy as np

from app.media import audio as A

SR = 48000


def test_timeline_places_clip(tmp_path):
    clip = tmp_path / "c.wav"
    A.write_wav(clip, A.sine(0.5))
    dub = A.build_dub_track([(1.0, clip)], 3.0)
    assert dub.shape[0] == 3 * SR
    nz = np.flatnonzero(np.abs(dub) > 1e-4)
    assert abs(nz[0] / SR - 1.0) < 0.001
    assert abs(dub[: SR - 10]).max() == 0


def test_timeline_truncates_past_end(tmp_path):
    clip = tmp_path / "c.wav"
    A.write_wav(clip, A.sine(2.0))
    dub = A.build_dub_track([(2.5, clip)], 3.0)
    assert dub.shape[0] == 3 * SR


def test_duck_gain_levels():
    n = 6 * SR
    g = A.duck_gain([(2.0, 4.0)], n, 0.15)
    assert abs(g[int(3.0 * SR)] - 0.15) < 0.01
    assert abs(g[int(0.5 * SR)] - 1.0) < 0.01
    assert abs(g[int(5.5 * SR)] - 1.0) < 0.01


def test_mix_peak_limited():
    n = SR
    orig = np.ones((n, 2), dtype=np.float32) * 0.9
    dub = np.ones(n, dtype=np.float32) * 0.9
    out = A.mix(orig, dub, np.ones(n, dtype=np.float32))
    assert np.abs(out).max() <= 0.98 + 1e-6
    assert out.shape == (n, 2)


def test_trim_silence(tmp_path):
    p = tmp_path / "s.wav"
    data = np.concatenate([np.zeros(SR), A.sine(1.0), np.zeros(SR)]).astype(np.float32)
    A.write_wav(p, data)
    d = A.trim_silence(p)
    assert 1.0 <= d <= 1.1
