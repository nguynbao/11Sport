from __future__ import annotations

import json
import subprocess

import pytest
from conftest import make_video

from app import storage
from app.config import get_settings
from app.constants import JobStatus
from app.db import SessionLocal
from app.models import Job, Segment
from app.pipeline.runner import run_phase_a, run_phase_b


def ffprobe_json(path):
    out = subprocess.run([get_settings().FFPROBE_BIN, "-v", "error", "-print_format", "json",
                          "-show_format", "-show_streams", str(path)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def create_job(video, lang="en", mode="hard", auto=False) -> str:
    with SessionLocal() as db:
        job = Job(group_id="g1", original_filename="Video thử.mp4", source_path="", target_lang=lang,
                  subtitle_mode=mode, auto_approve=auto, orig_volume=0.15, status=JobStatus.QUEUED)
        db.add(job)
        db.flush()
        dst = storage.job_dir(job.id) / f"source{video.suffix}"
        dst.write_bytes(video.read_bytes())
        job.source_path = storage.rel(dst)
        db.commit()
        return job.id


@pytest.mark.parametrize("mode,lang", [("hard", "en"), ("soft", "vi"), ("none", "en")])
def test_full_pipeline_with_review(env, sample_video, mode, lang):
    job_id = create_job(sample_video, lang, mode)
    run_phase_a(job_id)
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        assert job.status == JobStatus.REVIEW, job.error
        segs = db.query(Segment).filter_by(job_id=job_id).order_by(Segment.idx).all()
        assert len(segs) >= 2
        segs[0].translated_text = "Edited line" if lang == "en" else "修改后的句子"
        segs[0].edited = True
        job.status = JobStatus.RENDER_QUEUED
        db.commit()
    run_phase_b(job_id)
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        assert job.status == JobStatus.DONE, job.error
        assert job.progress == 100
        out = storage.abs_(job.output_path)
        assert out.exists()
        assert storage.abs_(job.srt_path).exists()
        segs = db.query(Segment).filter_by(job_id=job_id).all()
        assert all(s.fit_status != "pending" for s in segs)
    info = ffprobe_json(out)
    assert abs(float(info["format"]["duration"]) - 10.0) <= 0.3
    kinds = [s["codec_type"] for s in info["streams"]]
    assert kinds.count("audio") == 1
    assert kinds.count("subtitle") == (1 if mode == "soft" else 0)


def test_auto_approve_and_retry_from_checkpoint(env, sample_video):
    job_id = create_job(sample_video, "en", "soft", auto=True)
    run_phase_a(job_id)
    with SessionLocal() as db:
        assert db.get(Job, job_id).status == JobStatus.RENDER_QUEUED
    run_phase_b(job_id)
    tts_files = sorted(p.name for p in (storage.job_dir(job_id) / "tts").glob("*_raw.wav"))
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        assert job.status == JobStatus.DONE
        job.status = JobStatus.RENDER_QUEUED
        db.commit()
    run_phase_b(job_id)  # re-render must reuse TTS cache (no new raw files)
    assert sorted(p.name for p in (storage.job_dir(job_id) / "tts").glob("*_raw.wav")) == tts_files


def test_video_without_audio_fails(env, tmp_path):
    video = make_video(tmp_path / "silent.mp4", seconds=3, audio=False)
    job_id = create_job(video)
    run_phase_a(job_id)
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        assert job.status == JobStatus.FAILED
        assert job.failed_stage == "prepare"
        assert "không có âm thanh" in job.error


def test_vp9_webm_soft_reencodes(env, tmp_path):
    video = make_video(tmp_path / "v.webm", seconds=6, codec="libvpx-vp9")
    job_id = create_job(video, "en", "soft", auto=True)
    run_phase_a(job_id)
    run_phase_b(job_id)
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        assert job.status == JobStatus.DONE, job.error
        info = ffprobe_json(storage.abs_(job.output_path))
    vcodec = next(s["codec_name"] for s in info["streams"] if s["codec_type"] == "video")
    assert vcodec == "h264"


def test_asr_cache_shared_between_languages(env, sample_video):
    a = create_job(sample_video, "en")
    b = create_job(sample_video, "vi")
    run_phase_a(a)
    run_phase_a(b)
    assert len(list(storage.asr_cache_dir().glob("*.json"))) == 1
