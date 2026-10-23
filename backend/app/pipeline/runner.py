from __future__ import annotations

import logging
from collections.abc import Callable

from sqlalchemy.orm import Session

from ..constants import JobStatus
from ..db import SessionLocal
from ..errors import PipelineError
from ..models import Job
from . import stages

log = logging.getLogger(__name__)


def _fail(db: Session, job: Job, stage: str, e: BaseException) -> None:
    db.rollback()
    db.refresh(job)
    job.status = JobStatus.FAILED
    job.failed_stage = stage
    job.error = str(e) if isinstance(e, PipelineError) else f"Lỗi hệ thống: {e!r}"
    job.stage_message = "Lỗi"
    db.commit()


def _run(job_id: str, steps: list[tuple[str, Callable]]) -> None:
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if job is None:
            log.warning("job %s not found", job_id)
            return
        job.error = None
        job.failed_stage = None
        db.commit()
        carry = None
        for name, fn in steps:
            try:
                carry = fn(db, job, carry)
            except Exception as e:
                log.exception("job %s failed at %s", job_id, name)
                _fail(db, job, name, e)
                return
    finally:
        db.close()


def run_phase_a(job_id: str) -> None:
    _run(job_id, [
        ("prepare", lambda db, job, _: stages.stage_prepare(db, job)),
        ("transcribe", lambda db, job, _: stages.stage_transcribe(db, job)),
        ("translate", lambda db, job, asr: stages.stage_translate(db, job, asr)),
    ])


def run_phase_b(job_id: str) -> None:
    _run(job_id, [
        ("synthesize", lambda db, job, _: stages.stage_synthesize(db, job)),
        ("mix", lambda db, job, _: stages.stage_mix(db, job)),
        ("render", lambda db, job, _: stages.stage_render(db, job)),
    ])
