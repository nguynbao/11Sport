from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..config import get_settings
from ..constants import ISO639_2
from ..errors import PipelineError

log = logging.getLogger(__name__)


def _run(cmd: list[str], cwd: Path | None = None, what: str = "ffmpeg") -> subprocess.CompletedProcess:
    log.debug("run: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, check=False)
    except FileNotFoundError as e:
        raise PipelineError(f"Không tìm thấy {what} trên máy chủ ({cmd[0]})") from e
    if proc.returncode != 0:
        raise PipelineError(f"{what} lỗi (mã {proc.returncode}): {proc.stderr[-2000:]}")
    return proc


def ffmpeg_bin() -> str:
    return get_settings().FFMPEG_BIN


def ffprobe_bin() -> str:
    return get_settings().FFPROBE_BIN


@dataclass
class ProbeInfo:
    duration: float
    has_audio: bool
    has_video: bool
    width: int | None
    height: int | None
    video_codec: str | None


def probe(path: Path) -> ProbeInfo:
    proc = _run(
        [ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        what="ffprobe",
    )
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not (s.get("disposition") or {}).get("attached_pic")), None)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    duration = float((data.get("format") or {}).get("duration") or 0.0)
    return ProbeInfo(
        duration=duration,
        has_audio=has_audio,
        has_video=video is not None,
        width=int(video["width"]) if video and video.get("width") else None,
        height=int(video["height"]) if video and video.get("height") else None,
        video_codec=video.get("codec_name") if video else None,
    )


def audio_duration(path: Path) -> float:
    return probe(path).duration


def extract_asr_audio(src: Path, out: Path) -> None:
    _run([ffmpeg_bin(), "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
          "-c:a", "libmp3lame", "-b:a", "48k", str(out)])


def extract_orig_audio(src: Path, out: Path) -> None:
    _run([ffmpeg_bin(), "-y", "-i", str(src), "-vn", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(out)])


ADAM_AUDIO_FILTER = (
    "highpass=f=65,"
    "equalizer=f=115:t=q:w=1.2:g=2.2,"
    "equalizer=f=3000:t=q:w=1.4:g=2.0,"
    "acompressor=threshold=-18dB:ratio=3:attack=5:release=60:makeup=2.5dB,"
    "alimiter=limit=0.95"
)


def to_wav48_mono(src: Path, out: Path, adam_master: bool = False) -> None:
    cmd = [ffmpeg_bin(), "-y", "-i", str(src)]
    if adam_master:
        cmd += ["-filter:a", ADAM_AUDIO_FILTER]
    cmd += ["-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(out)]
    _run(cmd)


def atempo(src: Path, out: Path, factor: float) -> None:
    factor = min(2.0, max(0.5, factor))
    _run([ffmpeg_bin(), "-y", "-i", str(src), "-filter:a", f"atempo={factor:.4f}",
          "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(out)])


X264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]


def render(job_dir: Path, source_name: str, mode: str, lang: str, video_codec: str | None,
           duration: float | None = None) -> Path:
    """Render output.mp4 inside job_dir. All paths are relative to job_dir (cwd)."""
    lang3 = ISO639_2[lang]
    out = "output.mp4"
    base = [ffmpeg_bin(), "-y", "-i", source_name, "-i", "mixed.wav"]
    audio = ["-c:a", "aac", "-b:a", "192k", "-metadata:s:a:0", f"language={lang3}"]
    # NOTE: do not use -shortest: with a subtitle stream it cuts the video at the last cue.
    limit = ["-t", f"{duration:.3f}"] if duration else []
    tail = ["-movflags", "+faststart", *limit, out]
    can_copy = (video_codec or "").lower() in ("h264", "hevc")

    if mode == "hard":
        fonts = get_settings().fonts_path.resolve()
        fonts_str = str(fonts)
        if ":" in fonts_str or "'" in fonts_str:
            raise PipelineError("Đường dẫn thư mục font không hợp lệ (chứa ':' hoặc \"'\")")
        cmd = base + ["-map", "0:v:0", "-map", "1:a:0",
                      "-vf", f"subtitles=subs.ass:fontsdir={fonts_str}"] + X264 + audio + tail
    elif mode == "soft":
        vcodec = ["-c:v", "copy"] if can_copy else X264
        cmd = base + ["-i", "subs.srt", "-map", "0:v:0", "-map", "1:a:0", "-map", "2:s:0"] + vcodec + audio + [
            "-c:s", "mov_text", "-metadata:s:s:0", f"language={lang3}"] + tail
    else:
        vcodec = ["-c:v", "copy"] if can_copy else X264
        cmd = base + ["-map", "0:v:0", "-map", "1:a:0"] + vcodec + audio + tail
    # hevc in mp4 needs hvc1 tag for Apple players
    if can_copy and mode != "hard" and (video_codec or "").lower() == "hevc":
        cmd = cmd[:-len(tail)] + ["-tag:v", "hvc1"] + tail
    _run(cmd, cwd=job_dir)
    return job_dir / out
