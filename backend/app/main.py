from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import glossary, jobs, meta
from .config import get_settings
from .db import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_settings().validate_runtime()
    init_db()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="11Sport Dubber", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(meta.router, prefix="/api")
    app.include_router(jobs.router, prefix="/api")
    app.include_router(glossary.router, prefix="/api")
    return app


app = create_app()
