
## Bybit Futures Edge Lab 
is a Python project for researching and validating futures-trading hypotheses. It is **not** a production live-trading bot and does not place real orders.

The project supports an end-to-end research workflow:

1. collect public Bybit market data;
2. build datasets and trading features;
3. generate rule-based and ML-based signals;
4. evaluate ideas through backtests and walk-forward validation;
5. compare ML approaches with rule-based strategies;
6. simulate the full trade lifecycle in demo/paper mode;
7. review results through a REST API and Streamlit dashboard.

## Key Features

- **Public data ingestion** — historical candles, funding rates, open interest, long/short ratio, mark price, and other public Bybit data.
- **Feature engineering** — price, volume, volatility, funding, open-interest, microstructure, sentiment, liquidation, on-chain, and cross-market features.
- **Strategies** — mean reversion, funding arbitrage, order-book imbalance, cross-market momentum, and ML advisory signals.
- **Machine learning** — LightGBM experiments, model runs, feature importance, SHAP reports, and model-readiness checks.
- **Backtesting** — conservative taker-only assumptions, spread, slippage, funding, position limits, cooldowns, and walk-forward evaluation.
- **Paper/demo trading** — intents → approvals → simulated orders → fills → positions → close events → portfolio audit.
- **Risk controls** — trade-risk, daily-loss, drawdown, leverage, and exposure limits, plus a kill switch.
- **Interfaces** — FastAPI REST API, OpenAPI documentation at `/docs`, and a Streamlit dashboard.
- **Tests** — unit and integration coverage for ingestion, features, ML, backtests, risk, API, and PostgreSQL repositories.

## Safety Model

Live trading is deliberately blocked in v1:

- setting `REAL_TRADING_ENABLED=true` raises a configuration error;
- `RealBrokerInterface` rejects real-order placement;
- supported modes are `research`, `demo`, `paper`, and `real_advisory`;
- `real_advisory` can provide analytical recommendations but never executes orders.

> This repository is intended for research and education. It is not financial advice and does not guarantee performance.

## Technology Stack

- Python 3.12+
- FastAPI and Uvicorn
- PostgreSQL and SQLAlchemy
- Redis
- Pandas, NumPy, scikit-learn, LightGBM, SHAP
- Streamlit and Plotly
- Docker Compose
- Pytest, Ruff, Black, and mypy


## Quick Start

```bash
git clone <YOUR_REPOSITORY_URL>
cd BybitFuturesEdgeClean

python3.12 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[dev]"

cp .env.example .env
docker compose up -d postgres redis

uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

Once the API is running:

- Health check: `http://localhost:8000/health`
- Swagger / OpenAPI UI: `http://localhost:8000/docs`

## Seed Demo Data

Populate the local environment with demo market data:

```bash
curl -X POST "http://localhost:8000/demo/seed?symbols=BTCUSDT,ETHUSDT,SOLUSDT&timeframe=1m&rows=160"
```

## Run the Dashboard

```bash
streamlit run app/dashboard/streamlit_app.py
```

## Testing and Code Quality

Run unit tests:

```bash
pytest tests/unit -q
```

Run integration tests:

```bash
pytest tests/integration -q
```

Run linting, formatting checks, and static analysis:

```bash
ruff check .
black --check .
mypy app
```

## Current Limitations

- Real order placement is not supported by design.
- Public-market history should be collected through the public backfill endpoint or ingestion pipelines.
- The dashboard references commands such as `make seed-demo` and `make backfill-public`, but a `Makefile` is not currently included.
- Recommended next improvements: add a Makefile, CI/CD pipeline, Alembic migrations, dashboard screenshots, and `CONTRIBUTING.md`.

