from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .. import storage
from ..config import get_settings
from ..constants import LANG_SHORT_VI, JobStatus, normalize_lang
from ..errors import PipelineError
from ..media import audio as A
from ..media import ffmpeg as F
from ..media import subtitles as S
from ..models import GlossaryTerm, Job, Segment
from ..providers import get_asr, get_translator, get_tts, tts_voice_name
from ..providers.base import ASRResult, Glossary, TranslateItem
from ..text.budget import max_chars
from ..text.segmentation import build_segments
from .fit import fit_segment

log = logging.getLogger(__name__)


def set_stage(db: Session, job: Job, status: str, progress: int, message: str) -> None:
    job.status = status
    job.progress = progress
    job.stage_message = message
    db.commit()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_glossary(db: Session, target_lang: str | None) -> Glossary:
    terms = db.scalars(select(GlossaryTerm).order_by(GlossaryTerm.source_term)).all()
    if target_lang is None:
        return [(t.source_term, t.source_term) for t in terms]
    out = []
    for t in terms:
        tgt = getattr(t, target_lang, None)
        if t.source_term and tgt:
            out.append((t.source_term, tgt))
    return out


# ---------------------------------------------------------------- phase A
def stage_prepare(db: Session, job: Job) -> None:
    set_stage(db, job, JobStatus.PREPARING, 5, "Đang phân tích video")
    d = storage.job_dir(job.id)
    src = storage.abs_(job.source_path)
    if not src.exists():
        raise PipelineError("Không tìm thấy file video gốc")
    info = F.probe(src)
    if not info.has_video:
        raise PipelineError("File không có hình ảnh video")
    if not info.has_audio:
        raise PipelineError("Video không có âm thanh")
    if info.duration <= 0.5:
        raise PipelineError("Video quá ngắn")
    if info.duration > get_settings().MAX_DURATION_SEC:
        raise PipelineError(f"Video dài quá {get_settings().MAX_DURATION_SEC // 60} phút")
    job.duration_sec = info.duration
    job.width, job.height, job.video_codec = info.width, info.height, info.video_codec
    db.commit()
    if not (d / "audio_asr.mp3").exists():
        F.extract_asr_audio(src, d / "audio_asr.mp3.tmp.mp3")
        (d / "audio_asr.mp3.tmp.mp3").rename(d / "audio_asr.mp3")
    if not (d / "audio_orig.wav").exists():
        F.extract_orig_audio(src, d / "audio_orig.tmp.wav")
        (d / "audio_orig.tmp.wav").rename(d / "audio_orig.wav")


def stage_transcribe(db: Session, job: Job) -> ASRResult:
    set_stage(db, job, JobStatus.TRANSCRIBING, 20, "Đang nhận diện giọng nói")
    d = storage.job_dir(job.id)
    asr_json = d / "asr.json"
    if asr_json.exists():
        result = ASRResult.from_dict(json.loads(asr_json.read_text("utf-8")))
    else:
        s = get_settings()
        asr_tag = s.WHISPER_MODEL if s.PROVIDER_MODE == "free" else (
            s.OPENAI_ASR_MODEL if s.PROVIDER_MODE == "real" else "fake"
        )
        cache = storage.asr_cache_dir() / f"{_sha256(d / 'audio_asr.mp3')}_{asr_tag.replace('/', '_')}.json"
        if cache.exists():
            log.info("job %s: ASR cache hit", job.id)
            result = ASRResult.from_dict(json.loads(cache.read_text("utf-8")))
        else:
            terms = [s for s, _ in load_glossary(db, None)]
            prompt = ", ".join(terms)[:800] or None
            result = get_asr().transcribe(d / "audio_asr.mp3", prompt)
            cache.write_text(json.dumps(result.to_dict(), ensure_ascii=False), "utf-8")
        shutil.copyfile(cache, asr_json)
    job.source_lang = result.language or None
    db.commit()
    return result


def stage_translate(db: Session, job: Job, result: ASRResult) -> None:
    set_stage(db, job, JobStatus.TRANSLATING, 35, f"Đang dịch sang {LANG_SHORT_VI[job.target_lang]}")
    segs = build_segments(result)
    duration = job.duration_sec or segs[-1].end
    items: list[TranslateItem] = []
    for i, s in enumerate(segs):
        nxt = segs[i + 1].start if i + 1 < len(segs) else duration
        items.append(TranslateItem(i, s.text, max_chars(max(0.3, min(s.end, nxt) - s.start), job.target_lang)))
    if normalize_lang(job.source_lang) == job.target_lang:
        # Video already in the target language: keep the transcript (subtitles + re-voiced audio).
        translated = {it.id: it.text for it in items}
    else:
        translated = get_translator().translate(items, job.source_lang or "", job.target_lang,
                                                load_glossary(db, job.target_lang))
    db.execute(delete(Segment).where(Segment.job_id == job.id))
    for it, s in zip(items, segs, strict=True):
        db.add(Segment(job_id=job.id, idx=it.id, start=s.start, end=s.end, source_text=s.text,
                       translated_text=translated[it.id], max_chars=it.max_chars, fit_status="pending"))
    db.commit()
    db.refresh(job)
    if job.auto_approve:
        set_stage(db, job, JobStatus.RENDER_QUEUED, 40, "Đang chờ lồng tiếng")
    else:
        set_stage(db, job, JobStatus.REVIEW, 40, "Chờ duyệt bản dịch")


# ---------------------------------------------------------------- phase B
def stage_synthesize(db: Session, job: Job) -> None:
    segs = list(db.scalars(select(Segment).where(Segment.job_id == job.id).order_by(Segment.idx)))
    if not segs:
        raise PipelineError("Không có câu nào để lồng tiếng")
    n = len(segs)
    set_stage(db, job, JobStatus.SYNTHESIZING, 40, f"Đang tạo giọng đọc 0/{n}")
    tdir = storage.tts_dir(job.id)
    tts = get_tts(job.voice_gender)
    translator = get_translator()
    glossary = load_glossary(db, job.target_lang)
    voice = tts_voice_name(tts, job.target_lang, job.voice_gender)
    duration = job.duration_sec or (segs[-1].end + 1.0)

    for i, seg in enumerate(segs):
        idx = seg.idx
        counter = {"n": 0}

        def synth(text: str, rate: float, _idx: int = idx) -> tuple[Path, float]:
            key = hashlib.sha1(f"{text}|{voice}|{rate:.2f}|{tts.name}".encode()).hexdigest()[:10]
            raw = tdir / f"{_idx}_{key}_raw.wav"
            if not raw.exists():
                tmp = tdir / f"{_idx}_{key}_tmp.wav"
                tts.synthesize(text, job.target_lang, job.voice_gender, rate, tmp)
                A.trim_silence(tmp)
                tmp.rename(raw)
            else:
                log.debug("job %s seg %s: TTS cache hit", job.id, _idx)
            return raw, A.wav_duration(raw)

        def atempo(src: Path, factor: float, _idx: int = idx, _c: dict = counter) -> tuple[Path, float]:
            _c["n"] += 1
            out = tdir / f"{_idx}_work{_c['n']}.wav"
            F.atempo(src, out, factor)
            return out, A.wav_duration(out)

        def truncate(src: Path, max_sec: float, _idx: int = idx, _c: dict = counter) -> tuple[Path, float]:
            _c["n"] += 1
            out = tdir / f"{_idx}_work{_c['n']}.wav"
            shutil.copyfile(src, out)
            return out, A.fade_truncate(out, max_sec)

        def shorten(text: str, mc: int) -> str:
            return translator.shorten(text, job.target_lang, mc, glossary)

        next_start = segs[i + 1].start if i + 1 < n else duration + 0.05
        res = fit_segment(seg.translated_text, seg.start, next_start, job.target_lang,
                          synth, shorten, atempo, truncate, orig_end=seg.end)
        fit_path = tdir / f"{idx}_fit.wav"
        shutil.copyfile(res.audio, fit_path)
        for p in tdir.glob(f"{idx}_work*.wav"):
            p.unlink(missing_ok=True)
        if res.shortened:
            seg.translated_text = res.text
        seg.tts_duration = round(res.duration, 3)
        seg.placed_start = seg.start
        seg.placed_end = round(seg.start + res.duration, 3)
        seg.speed = res.speed
        seg.fit_status = res.status
        job.progress = 40 + int(40 * (i + 1) / n)
        job.stage_message = f"Đang tạo giọng đọc {i + 1}/{n}"
        db.commit()


def stage_mix(db: Session, job: Job) -> None:
    set_stage(db, job, JobStatus.MIXING, 85, "Đang trộn âm thanh")
    d = storage.job_dir(job.id)
    segs = list(db.scalars(select(Segment).where(Segment.job_id == job.id).order_by(Segment.idx)))
    duration = job.duration_sec or 0.0
    placements = [(s.placed_start or s.start, d / "tts" / f"{s.idx}_fit.wav") for s in segs]
    dub = A.build_dub_track(placements, duration)
    A.write_wav(d / "dub_track.wav", dub)
    orig, sr = A.read_wav(d / "audio_orig.wav")
    if sr != 48000:
        raise PipelineError("Audio gốc sai sample rate")
    regions = [(s.placed_start or s.start, s.placed_end or s.end) for s in segs]
    gain = A.duck_gain(regions, dub.shape[0], job.orig_volume)
    A.write_wav(d / "mixed.wav", A.mix(orig, dub, gain))


def stage_render(db: Session, job: Job) -> None:
    set_stage(db, job, JobStatus.RENDERING, 92, "Đang xuất video")
    d = storage.job_dir(job.id)
    segs = list(db.scalars(select(Segment).where(Segment.job_id == job.id).order_by(Segment.idx)))
    sources = [S.CueSource(s.placed_start if s.placed_start is not None else s.start, s.end,
                           s.placed_end if s.placed_end is not None else s.end, s.translated_text) for s in segs]
    w, h = job.width or 1280, job.height or 720
    cues = S.build_cues(sources, job.target_lang, job.duration_sec, w, h)
    S.write_srt(cues, d / "subs.srt")
    S.write_ass(cues, d / "subs.ass", w, h, job.target_lang)
    src_name = Path(job.source_path).name
    (d / "output.mp4").unlink(missing_ok=True)
    out = F.render(d, src_name, job.subtitle_mode, job.target_lang, job.video_codec, job.duration_sec)
    job.output_path = storage.rel(out)
    job.srt_path = storage.rel(d / "subs.srt")
    set_stage(db, job, JobStatus.DONE, 100, "Hoàn tất")
