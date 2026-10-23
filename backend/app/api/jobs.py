from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import storage
from ..config import get_settings
from ..constants import ALLOWED_EXTS, GENDERS, RUNNING_STATUSES, SUBTITLE_MODES, TARGET_LANGS, JobStatus
from ..db import get_db
from ..models import Job, Segment, new_id
from ..schemas import JobList, JobOut, SegmentOut, SegmentUpdate

router = APIRouter(prefix="/jobs", tags=["jobs"])
DB = Annotated[Session, Depends(get_db)]


def _job_out(db: Session, job: Job) -> JobOut:
    out = JobOut.model_validate(job)
    out.has_output = bool(job.output_path) and storage.abs_(job.output_path).exists()
    out.segment_count = db.scalar(select(func.count()).select_from(Segment).where(Segment.job_id == job.id)) or 0
    out.overflow_count = db.scalar(
        select(func.count()).select_from(Segment).where(Segment.job_id == job.id, Segment.fit_status == "overflow")
    ) or 0
    return out


def _get(db: Session, job_id: str) -> Job:
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Không tìm thấy job")
    return job


def _parse_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


@router.post("", response_model=list[JobOut])
def create_jobs(
    db: DB,
    file: Annotated[UploadFile, File()],
    target_langs: Annotated[str, Form()] = "en",
    voice_gender: Annotated[str, Form()] = "female",
    subtitle_mode: Annotated[str, Form()] = "hard",
    auto_approve: Annotated[str, Form()] = "false",
    orig_volume: Annotated[float | None, Form()] = None,
) -> list[JobOut]:
    s = get_settings()
    filename = file.filename or "video"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"Định dạng không hỗ trợ. Chỉ chấp nhận: {', '.join(sorted(ALLOWED_EXTS))}")
    langs = [x.strip().lower() for x in target_langs.split(",") if x.strip()]
    if not langs or any(x not in TARGET_LANGS for x in langs) or len(set(langs)) != len(langs):
        raise HTTPException(400, "Ngôn ngữ đích không hợp lệ (chỉ hỗ trợ en, vi và không trùng lặp)")
    if voice_gender not in GENDERS:
        raise HTTPException(400, "Giọng đọc không hợp lệ")
    if subtitle_mode not in SUBTITLE_MODES:
        raise HTTPException(400, "Chế độ phụ đề không hợp lệ")
    vol = s.DEFAULT_ORIG_VOLUME if orig_volume is None else float(orig_volume)
    if not 0.0 <= vol <= 0.5:
        raise HTTPException(400, "Âm lượng tiếng gốc phải trong khoảng 0–0.5")

    # Stream upload to a temp file with size limit
    tmp_dir = storage.storage_root() / "uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / f"{uuid.uuid4().hex}{ext}"
    limit = s.MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    try:
        with open(tmp, "wb") as out:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"File vượt quá {s.MAX_UPLOAD_MB} MB")
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, "File rỗng")

        group = new_id()
        jobs: list[Job] = []
        for lang in langs:
            job_id = new_id()
            dst = storage.job_dir(job_id) / f"source{ext}"
            shutil.copyfile(tmp, dst)
            job = Job(id=job_id, group_id=group, original_filename=filename, source_path=storage.rel(dst),
                      target_lang=lang, voice_gender=voice_gender, subtitle_mode=subtitle_mode,
                      auto_approve=_parse_bool(auto_approve), orig_volume=vol, status=JobStatus.QUEUED,
                      progress=0, stage_message="Đang chờ xử lý")
            db.add(job)
            jobs.append(job)
        db.commit()
    finally:
        tmp.unlink(missing_ok=True)
    return [_job_out(db, j) for j in jobs]


@router.get("", response_model=JobList)
def list_jobs(db: DB, limit: int = 50, offset: int = 0) -> JobList:
    limit = max(1, min(limit, 200))
    total = db.scalar(select(func.count()).select_from(Job)) or 0
    rows = db.scalars(select(Job).order_by(Job.created_at.desc()).limit(limit).offset(offset)).all()
    return JobList(items=[_job_out(db, j) for j in rows], total=total)


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: DB) -> JobOut:
    return _job_out(db, _get(db, job_id))


@router.get("/{job_id}/segments", response_model=list[SegmentOut])
def get_segments(job_id: str, db: DB) -> list[Segment]:
    _get(db, job_id)
    return list(db.scalars(select(Segment).where(Segment.job_id == job_id).order_by(Segment.idx)))


@router.put("/{job_id}/segments", response_model=list[SegmentOut])
def update_segments(job_id: str, body: SegmentUpdate, db: DB) -> list[Segment]:
    job = _get(db, job_id)
    if job.status != JobStatus.REVIEW:
        raise HTTPException(409, "Chỉ sửa được bản dịch khi job đang chờ duyệt")
    segs = {s.idx: s for s in db.scalars(select(Segment).where(Segment.job_id == job_id))}
    for item in body.items:
        seg = segs.get(item.idx)
        if seg is None:
            raise HTTPException(400, f"Không có câu số {item.idx}")
        text = item.translated_text.strip()
        if not text:
            raise HTTPException(400, f"Bản dịch câu {item.idx} không được để trống")
        if text != seg.translated_text:
            seg.translated_text = text
            seg.edited = True
            seg.fit_status = "pending"
    db.commit()
    return list(db.scalars(select(Segment).where(Segment.job_id == job_id).order_by(Segment.idx)))


@router.post("/{job_id}/approve", response_model=JobOut)
def approve(job_id: str, db: DB) -> JobOut:
    job = _get(db, job_id)
    if job.status != JobStatus.REVIEW:
        raise HTTPException(409, "Job không ở trạng thái chờ duyệt")
    job.status = JobStatus.RENDER_QUEUED
    job.stage_message = "Đang chờ lồng tiếng"
    db.commit()
    return _job_out(db, job)


@router.post("/{job_id}/reopen", response_model=JobOut)
def reopen(job_id: str, db: DB) -> JobOut:
    job = _get(db, job_id)
    has_segs = db.scalar(select(func.count()).select_from(Segment).where(Segment.job_id == job_id)) or 0
    if job.status not in (JobStatus.DONE, JobStatus.FAILED) or not has_segs:
        raise HTTPException(409, "Chỉ mở lại được job đã hoàn tất/lỗi và đã có bản dịch")
    job.status = JobStatus.REVIEW
    job.progress = 40
    job.stage_message = "Chờ duyệt bản dịch"
    job.error = None
    job.failed_stage = None
    db.commit()
    return _job_out(db, job)


@router.post("/{job_id}/retry", response_model=JobOut)
def retry(job_id: str, db: DB) -> JobOut:
    job = _get(db, job_id)
    if job.status != JobStatus.FAILED:
        raise HTTPException(409, "Chỉ thử lại được job bị lỗi")
    has_segs = db.scalar(select(func.count()).select_from(Segment).where(Segment.job_id == job_id)) or 0
    job.status = JobStatus.RENDER_QUEUED if has_segs else JobStatus.QUEUED
    job.error = None
    job.failed_stage = None
    job.stage_message = "Đang chờ xử lý lại"
    db.commit()
    return _job_out(db, job)


@router.delete("/{job_id}")
def delete_job(job_id: str, db: DB) -> dict:
    job = _get(db, job_id)
    if job.status in {s.value for s in RUNNING_STATUSES}:
        raise HTTPException(409, "Job đang xử lý, không thể xóa")
    db.delete(job)
    db.commit()
    storage.delete_job_dir(job_id)
    return {"ok": True}


@router.get("/{job_id}/files/{kind}")
def get_file(job_id: str, kind: str, db: DB) -> FileResponse:
    job = _get(db, job_id)
    stem = Path(job.original_filename).stem or "video"
    if kind == "source":
        rel, name, media = job.source_path, job.original_filename, None
    elif kind == "output":
        rel, name, media = job.output_path, f"{stem}_{job.target_lang}.mp4", "video/mp4"
    elif kind == "srt":
        rel, name, media = job.srt_path, f"{stem}_{job.target_lang}.srt", "application/x-subrip"
    else:
        raise HTTPException(400, "Loại file không hợp lệ")
    if not rel or not storage.abs_(rel).exists():
        raise HTTPException(404, "File chưa sẵn sàng")
    return FileResponse(storage.abs_(rel), filename=name, media_type=media,
                        content_disposition_type="inline" if kind != "srt" else "attachment")
