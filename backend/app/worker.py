from __future__ import annotations

import logging
import shutil
import signal
import time
from datetime import timedelta

from sqlalchemy import select, update

from . import storage
from .config import get_settings
from .constants import PHASE_A_RUNNING, PHASE_B_RUNNING, JobStatus
from .db import SessionLocal, init_db
from .models import Job, utcnow
from .pipeline.runner import run_phase_a, run_phase_b

log = logging.getLogger("worker")

_stop = False


def _handle_signal(signum, _frame):
    global _stop
    log.info("signal %s received: finishing current job then exiting", signum)
    _stop = True


def recover_stuck_jobs() -> None:
    with SessionLocal() as db:
        a = db.execute(update(Job).where(Job.status.in_([s.value for s in PHASE_A_RUNNING]))
                       .values(status=JobStatus.QUEUED, stage_message="Đang chờ (khôi phục)"))
        b = db.execute(update(Job).where(Job.status.in_([s.value for s in PHASE_B_RUNNING]))
                       .values(status=JobStatus.RENDER_QUEUED, stage_message="Đang chờ (khôi phục)"))
        db.commit()
        if a.rowcount or b.rowcount:
            log.info("recovered %s phase-A and %s phase-B jobs", a.rowcount, b.rowcount)


def claim_next() -> tuple[str, str] | None:
    """Atomically claim the oldest queued job. Returns (job_id, phase)."""
    with SessionLocal() as db:
        candidates = db.execute(
            select(Job.id, Job.status)
            .where(Job.status.in_([JobStatus.QUEUED, JobStatus.RENDER_QUEUED]))
            .order_by(Job.created_at)
            .limit(5)
        ).all()
        for job_id, status in candidates:
            nxt = JobStatus.PREPARING if status == JobStatus.QUEUED else JobStatus.SYNTHESIZING
            res = db.execute(update(Job).where(Job.id == job_id, Job.status == status).values(status=nxt))
            db.commit()
            if res.rowcount == 1:
                return job_id, ("A" if status == JobStatus.QUEUED else "B")
    return None


def process_one() -> bool:
    claimed = claim_next()
    if not claimed:
        return False
    job_id, phase = claimed
    log.info("processing job %s phase %s", job_id, phase)
    (run_phase_a if phase == "A" else run_phase_b)(job_id)
    return True


def cleanup() -> None:
    days = get_settings().RETENTION_DAYS
    cutoff = utcnow() - timedelta(days=days)
    with SessionLocal() as db:
        old = db.scalars(select(Job).where(Job.created_at < cutoff)).all()
        for job in old:
            if job.status in {s.value for s in PHASE_A_RUNNING | PHASE_B_RUNNING}:
                continue
            storage.delete_job_dir(job.id)
            db.delete(job)
        db.commit()
        if old:
            log.info("cleanup: removed %s old jobs", len(old))
    cache = storage.asr_cache_dir()
    limit = time.time() - days * 86400
    for p in cache.glob("*.json"):
        if p.stat().st_mtime < limit:
            p.unlink(missing_ok=True)
    # orphan job dirs
    jobs_root = storage.storage_root() / "jobs"
    if jobs_root.exists():
        with SessionLocal() as db:
            ids = set(db.scalars(select(Job.id)).all())
        for d in jobs_root.iterdir():
            if d.is_dir() and d.name not in ids and d.stat().st_mtime < limit:
                shutil.rmtree(d, ignore_errors=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = get_settings()
    s.validate_runtime()
    init_db()
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    recover_stuck_jobs()
    log.info("worker started (provider_mode=%s, tts=%s)", s.PROVIDER_MODE, s.TTS_PROVIDER)
    last_cleanup = 0.0
    while not _stop:
        if time.time() - last_cleanup > 3600:
            try:
                cleanup()
            except Exception:
                log.exception("cleanup failed")
            last_cleanup = time.time()
        try:
            worked = process_one()
        except Exception:
            log.exception("worker loop error")
            worked = False
        if not worked:
            time.sleep(s.WORKER_POLL_SEC)
    log.info("worker stopped")


if __name__ == "__main__":
    main()
