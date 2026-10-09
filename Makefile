PYTHON ?= python3
SYMBOLS ?= BTCUSDT,ETHUSDT,SOLUSDT
SYMBOL ?= BTCUSDT
TIMEFRAME ?= 1m
ROWS ?= 160
HOURS ?= 24

.PHONY: setup services api dashboard seed-demo backfill-public test test-integration lint format-check typecheck check

setup:
	$(PYTHON) -m pip install -e ".[dev]"

services:
	docker compose up -d postgres redis

api:
	$(PYTHON) -m uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	$(PYTHON) -m streamlit run app/dashboard/streamlit_app.py

seed-demo:
	$(PYTHON) -m app.pipelines.demo_seed --symbols $(SYMBOLS) --timeframe $(TIMEFRAME) --rows $(ROWS)

backfill-public:
	$(PYTHON) -m app.ingestion.public_backfill --symbol $(SYMBOL) --timeframe $(TIMEFRAME) --hours $(HOURS)

test:
	$(PYTHON) -m pytest tests/unit -q

test-integration:
	$(PYTHON) -m pytest tests/integration -q

lint:
	$(PYTHON) -m ruff check .

format-check:
	$(PYTHON) -m black --check .

typecheck:
	$(PYTHON) -m mypy app

check: lint format-check typecheck test
