export PATH := $(HOME)/.local/bin:$(PATH)

.PHONY: setup api worker web test lint sample build

setup:
	cd backend && uv sync && (test -f .env || cp .env.example .env)
	cd frontend && npm install

api:
	cd backend && uv run uvicorn app.main:app --reload --port 8000

worker:
	cd backend && uv run python -m app.worker

web:
	cd frontend && npm run dev

test:
	cd backend && uv run pytest -q

lint:
	cd backend && uv run ruff check app tests

build:
	cd frontend && npm run build

sample:
	mkdir -p storage && cd backend && bash scripts/make_sample_video.sh ../storage/sample.mp4
