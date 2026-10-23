from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    group_id: str
    created_at: datetime
    updated_at: datetime
    original_filename: str
    duration_sec: float | None
    width: int | None
    height: int | None
    video_codec: str | None
    source_lang: str | None
    target_lang: str
    voice_gender: str
    subtitle_mode: str
    auto_approve: bool
    orig_volume: float
    status: str
    progress: int
    stage_message: str | None
    error: str | None
    failed_stage: str | None
    has_output: bool = False
    segment_count: int = 0
    overflow_count: int = 0


class JobList(BaseModel):
    items: list[JobOut]
    total: int


class SegmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    job_id: str
    idx: int
    start: float
    end: float
    source_text: str
    translated_text: str
    max_chars: int
    edited: bool
    tts_duration: float | None
    placed_start: float | None
    placed_end: float | None
    speed: float | None
    fit_status: str


class SegmentUpdateItem(BaseModel):
    idx: int
    translated_text: str = Field(min_length=1, max_length=2000)


class SegmentUpdate(BaseModel):
    items: list[SegmentUpdateItem]


class GlossaryIn(BaseModel):
    source_term: str = Field(min_length=1, max_length=255)
    en: str | None = None
    vi: str | None = None
    note: str | None = None


class GlossaryOut(GlossaryIn):
    model_config = ConfigDict(from_attributes=True)
    id: int
