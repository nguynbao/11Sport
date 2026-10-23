from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .constants import JobStatus
from .db import Base


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=new_id)
    group_id: Mapped[str] = mapped_column(String(12), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    original_filename: Mapped[str] = mapped_column(String(512))
    source_path: Mapped[str] = mapped_column(String(512))
    duration_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    video_codec: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_lang: Mapped[str | None] = mapped_column(String(64), nullable=True)
    target_lang: Mapped[str] = mapped_column(String(8))
    voice_gender: Mapped[str] = mapped_column(String(8), default="female")
    subtitle_mode: Mapped[str] = mapped_column(String(8), default="hard")
    auto_approve: Mapped[bool] = mapped_column(Boolean, default=False)
    orig_volume: Mapped[float] = mapped_column(Float, default=0.05)
    status: Mapped[str] = mapped_column(String(16), default=JobStatus.QUEUED, index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    stage_message: Mapped[str | None] = mapped_column(String(255), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    failed_stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    output_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    srt_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    segments: Mapped[list[Segment]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="Segment.idx", passive_deletes=True
    )


class Segment(Base):
    __tablename__ = "segments"
    __table_args__ = (UniqueConstraint("job_id", "idx", name="uq_segment_job_idx"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    idx: Mapped[int] = mapped_column(Integer)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    source_text: Mapped[str] = mapped_column(Text)
    translated_text: Mapped[str] = mapped_column(Text, default="")
    max_chars: Mapped[int] = mapped_column(Integer, default=0)
    edited: Mapped[bool] = mapped_column(Boolean, default=False)
    tts_duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    placed_start: Mapped[float | None] = mapped_column(Float, nullable=True)
    placed_end: Mapped[float | None] = mapped_column(Float, nullable=True)
    speed: Mapped[float | None] = mapped_column(Float, nullable=True)
    fit_status: Mapped[str] = mapped_column(String(16), default="pending")

    job: Mapped[Job] = relationship(back_populates="segments")


class GlossaryTerm(Base):
    __tablename__ = "glossary_terms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_term: Mapped[str] = mapped_column(String(255), unique=True)
    en: Mapped[str | None] = mapped_column(String(255), nullable=True)
    vi: Mapped[str | None] = mapped_column(String(255), nullable=True)
    note: Mapped[str | None] = mapped_column(String(512), nullable=True)
