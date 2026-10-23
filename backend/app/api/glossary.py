from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import GlossaryTerm
from ..schemas import GlossaryIn, GlossaryOut

router = APIRouter(prefix="/glossary", tags=["glossary"])
DB = Annotated[Session, Depends(get_db)]


def _clean(body: GlossaryIn) -> dict:
    d = body.model_dump()
    d["source_term"] = d["source_term"].strip()
    for k in ("en", "vi", "note"):
        d[k] = (d[k] or "").strip() or None
    return d


@router.get("", response_model=list[GlossaryOut])
def list_terms(db: DB) -> list[GlossaryTerm]:
    return list(db.scalars(select(GlossaryTerm).order_by(GlossaryTerm.source_term)))


@router.post("", response_model=GlossaryOut)
def create_term(body: GlossaryIn, db: DB) -> GlossaryTerm:
    term = GlossaryTerm(**_clean(body))
    db.add(term)
    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(409, "Thuật ngữ đã tồn tại") from e
    return term


@router.put("/{term_id}", response_model=GlossaryOut)
def update_term(term_id: int, body: GlossaryIn, db: DB) -> GlossaryTerm:
    term = db.get(GlossaryTerm, term_id)
    if not term:
        raise HTTPException(404, "Không tìm thấy thuật ngữ")
    for k, v in _clean(body).items():
        setattr(term, k, v)
    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(409, "Thuật ngữ đã tồn tại") from e
    return term


@router.delete("/{term_id}")
def delete_term(term_id: int, db: DB) -> dict:
    term = db.get(GlossaryTerm, term_id)
    if not term:
        raise HTTPException(404, "Không tìm thấy thuật ngữ")
    db.delete(term)
    db.commit()
    return {"ok": True}
