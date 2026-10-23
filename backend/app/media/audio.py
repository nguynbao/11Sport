from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf

from ..constants import SAMPLE_RATE


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return data, sr


def write_wav(path: Path, data: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    sf.write(str(path), np.clip(data, -1.0, 1.0), sr, subtype="PCM_16")


def wav_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def trim_silence(path: Path, threshold_db: float = -45.0, pad_ms: int = 30) -> float:
    """Trim leading/trailing silence in place. Returns new duration in seconds."""
    data, sr = read_wav(path)
    mono = data if data.ndim == 1 else data.mean(axis=1)
    if mono.size == 0:
        return 0.0
    thr = 10 ** (threshold_db / 20.0)
    idx = np.flatnonzero(np.abs(mono) >= thr)
    if idx.size == 0:
        return mono.size / sr
    pad = int(sr * pad_ms / 1000)
    a = max(0, idx[0] - pad)
    b = min(mono.size, idx[-1] + 1 + pad)
    write_wav(path, data[a:b], sr)
    return (b - a) / sr


def fade_truncate(path: Path, max_sec: float, fade_ms: int = 80) -> float:
    data, sr = read_wav(path)
    n = int(max_sec * sr)
    if data.shape[0] <= n:
        return data.shape[0] / sr
    data = data[:n].copy()
    f = min(n, int(sr * fade_ms / 1000))
    if f > 0:
        ramp = np.linspace(1.0, 0.0, f, dtype=np.float32)
        data[-f:] = data[-f:] * (ramp if data.ndim == 1 else ramp[:, None])
    write_wav(path, data, sr)
    return n / sr


def build_dub_track(placements: Sequence[tuple[float, Path]], duration_sec: float,
                    sr: int = SAMPLE_RATE) -> np.ndarray:
    total = round(duration_sec * sr)
    dub = np.zeros(total, dtype=np.float32)
    for start, path in placements:
        clip, csr = read_wav(path)
        if clip.ndim > 1:
            clip = clip.mean(axis=1)
        assert csr == sr, f"TTS sample rate {csr} != {sr}"
        s = round(start * sr)
        if s >= total:
            continue
        e = min(total, s + clip.shape[0])
        dub[s:e] += clip[: e - s]
    return dub


def duck_gain(regions: Sequence[tuple[float, float]], n_samples: int, orig_volume: float,
              sr: int = SAMPLE_RATE, pad: float = 0.10, smooth_ms: int = 150) -> np.ndarray:
    """Gain envelope: 1 outside voice-over regions, orig_volume inside, smoothed ramps."""
    dur = n_samples / sr
    n1k = int(np.ceil(dur * 1000)) + 1
    mask = np.zeros(n1k, dtype=np.float32)
    for a, b in regions:
        ia = max(0, int((a - pad) * 1000))
        ib = min(n1k, int(np.ceil((b + pad) * 1000)))
        if ib > ia:
            mask[ia:ib] = 1.0
    if smooth_ms > 1:
        kernel = np.ones(smooth_ms, dtype=np.float32) / smooth_ms
        mask = np.convolve(mask, kernel, mode="same")
    gain_1k = 1.0 - mask * (1.0 - orig_volume)
    t_1k = np.arange(n1k, dtype=np.float64) / 1000.0
    t = np.arange(n_samples, dtype=np.float64) / sr
    return np.interp(t, t_1k, gain_1k).astype(np.float32)


def mix(orig: np.ndarray, dub: np.ndarray, gain: np.ndarray) -> np.ndarray:
    n = dub.shape[0]
    if orig.ndim == 1:
        orig = np.stack([orig, orig], axis=1)
    if orig.shape[0] < n:
        orig = np.concatenate([orig, np.zeros((n - orig.shape[0], orig.shape[1]), dtype=np.float32)])
    orig = orig[:n]
    out = orig * gain[:, None] + dub[:, None]
    peak = float(np.abs(out).max()) if out.size else 0.0
    if peak > 0.98:
        out *= 0.98 / peak
    return out.astype(np.float32)


def sine(duration: float, freq: float = 220.0, amp: float = 0.3, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
