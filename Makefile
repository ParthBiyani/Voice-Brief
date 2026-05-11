.DEFAULT_GOAL := help
PY := python
COMPOSE := docker compose

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with all extras
	$(PY) -m pip install -e ".[ml,llm,docs,tts,dev]"

up:  ## Start Postgres, Qdrant, Redis, MinIO
	$(COMPOSE) up -d --wait

down:  ## Stop VoiceBrief services (leaves volumes intact)
	$(COMPOSE) stop

migrate:  ## Apply database migrations
	alembic upgrade head

seed:  ## Load the source registry from config/sources.yaml
	$(PY) -m voicebrief.cli sources sync

ingest:  ## Run one ingestion pass across all enabled sources
	$(PY) -m voicebrief.cli ingest run

brief:  ## Generate today's episode end to end
	$(PY) -m voicebrief.cli brief generate

api:  ## Serve the API with reload
	uvicorn voicebrief.api.main:app --reload --port 8000

web:  ## Serve the React app
	cd web && npm run dev

test:  ## Run the unit test suite
	pytest -q -m "not integration and not network and not slow"

test-all:  ## Run every test including integration
	pytest -q

lint:  ## Lint and type-check
	ruff check src tests
	ruff format --check src tests
	mypy src

fmt:  ## Auto-format
	ruff check --fix src tests
	ruff format src tests

eval:  ## Run the evaluation harness and diff against the last recorded run
	$(PY) -m voicebrief.evalkit.run --compare

.PHONY: help install up down migrate seed ingest brief api web test test-all lint fmt eval
