CREATE TABLE IF NOT EXISTS raw_bars (
  id BIGSERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  timeframe TEXT NOT NULL,
  event_time TIMESTAMPTZ NOT NULL,
  fetch_time TIMESTAMPTZ NOT NULL,
  open NUMERIC NOT NULL,
  high NUMERIC NOT NULL,
  low NUMERIC NOT NULL,
  close NUMERIC NOT NULL,
  volume NUMERIC NOT NULL,
  source TEXT NOT NULL,
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE(symbol, timeframe, event_time, source)
);

CREATE TABLE IF NOT EXISTS raw_mark_bars (LIKE raw_bars INCLUDING ALL);
CREATE TABLE IF NOT EXISTS raw_index_bars (LIKE raw_bars INCLUDING ALL);

CREATE TABLE IF NOT EXISTS raw_funding (
  id BIGSERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  event_time TIMESTAMPTZ NOT NULL,
  fetch_time TIMESTAMPTZ NOT NULL,
  funding_rate NUMERIC NOT NULL,
  source TEXT NOT NULL,
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE UNIQUE INDEX IF NOT EXISTS raw_funding_symbol_event_source_idx
  ON raw_funding (symbol, event_time, source);

CREATE TABLE IF NOT EXISTS raw_open_interest (
  id BIGSERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  event_time TIMESTAMPTZ NOT NULL,
  fetch_time TIMESTAMPTZ NOT NULL,
  open_interest NUMERIC NOT NULL,
  source TEXT NOT NULL,
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE UNIQUE INDEX IF NOT EXISTS raw_open_interest_symbol_event_source_idx
  ON raw_open_interest (symbol, event_time, source);

CREATE TABLE IF NOT EXISTS raw_long_short_ratio (
  id BIGSERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  event_time TIMESTAMPTZ NOT NULL,
  fetch_time TIMESTAMPTZ NOT NULL,
  long_short_ratio NUMERIC NOT NULL,
  source TEXT NOT NULL,
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE UNIQUE INDEX IF NOT EXISTS raw_long_short_ratio_symbol_event_source_idx
  ON raw_long_short_ratio (symbol, event_time, source);

CREATE TABLE IF NOT EXISTS raw_trades (
  id BIGSERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  event_time TIMESTAMPTZ NOT NULL,
  fetch_time TIMESTAMPTZ NOT NULL,
  price NUMERIC NOT NULL,
  quantity NUMERIC NOT NULL,
  side TEXT NOT NULL,
  source TEXT NOT NULL,
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS raw_orderbook (
  id BIGSERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  event_time TIMESTAMPTZ NOT NULL,
  fetch_time TIMESTAMPTZ NOT NULL,
  bids_json JSONB NOT NULL,
  asks_json JSONB NOT NULL,
  source TEXT NOT NULL,
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS raw_liquidations (
  id BIGSERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  event_time TIMESTAMPTZ NOT NULL,
  fetch_time TIMESTAMPTZ NOT NULL,
  side TEXT NOT NULL,
  price NUMERIC NOT NULL,
  quantity NUMERIC NOT NULL,
  source TEXT NOT NULL,
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS ext_market_bars (LIKE raw_bars INCLUDING ALL);

CREATE TABLE IF NOT EXISTS onchain_metrics (
  id BIGSERIAL PRIMARY KEY,
  provider TEXT NOT NULL,
  metric TEXT NOT NULL,
  asset TEXT NOT NULL,
  event_time TIMESTAMPTZ NOT NULL,
  publish_time TIMESTAMPTZ NOT NULL,
  fetch_time TIMESTAMPTZ NOT NULL,
  value NUMERIC NOT NULL,
  source TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS news_items (
  id BIGSERIAL PRIMARY KEY,
  provider TEXT NOT NULL,
  item_id TEXT NOT NULL,
  headline TEXT NOT NULL,
  body TEXT NOT NULL DEFAULT '',
  published_at TIMESTAMPTZ NOT NULL,
  ingested_at TIMESTAMPTZ NOT NULL,
  event_time TIMESTAMPTZ NOT NULL,
  publish_time TIMESTAMPTZ NOT NULL,
  sentiment_raw NUMERIC,
  entities_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  source TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feature_rows (
  id BIGSERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  event_time TIMESTAMPTZ NOT NULL,
  feature_set_version TEXT NOT NULL,
  features_json JSONB NOT NULL,
  source_rows_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS label_rows (
  id BIGSERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  event_time TIMESTAMPTZ NOT NULL,
  labels_json JSONB NOT NULL,
  cost_assumptions_json JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS model_runs (
  model_run_id TEXT PRIMARY KEY,
  symbol TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL,
  feature_set_version TEXT NOT NULL,
  metrics_json JSONB NOT NULL,
  params_json JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS model_artifacts (
  artifact_id TEXT PRIMARY KEY,
  model_run_id TEXT NOT NULL REFERENCES model_runs(model_run_id),
  path TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy_comparison_snapshots (
  snapshot_id TEXT PRIMARY KEY,
  model_run_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  timeframe TEXT NOT NULL,
  feature_set_version TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL,
  comparison_json JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS strategy_comparison_snapshots_model_created_idx
  ON strategy_comparison_snapshots (model_run_id, created_at DESC);

CREATE INDEX IF NOT EXISTS strategy_comparison_snapshots_symbol_tf_created_idx
  ON strategy_comparison_snapshots (symbol, timeframe, created_at DESC);

CREATE TABLE IF NOT EXISTS predictions (
  prediction_id TEXT PRIMARY KEY,
  model_run_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  event_time TIMESTAMPTZ NOT NULL,
  probabilities_json JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
  signal_id TEXT PRIMARY KEY,
  symbol TEXT NOT NULL,
  event_time TIMESTAMPTZ NOT NULL,
  strategy_name TEXT NOT NULL,
  side TEXT NOT NULL,
  score NUMERIC NOT NULL,
  confidence NUMERIC NOT NULL,
  rationale_json JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS signal_audit_logs (
  audit_id TEXT PRIMARY KEY,
  signal_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  event_time TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL,
  strategy_name TEXT NOT NULL,
  model_run_id TEXT,
  feature_set_version TEXT NOT NULL,
  feature_snapshot_json JSONB NOT NULL,
  rationale_json JSONB NOT NULL,
  risk_checks_json JSONB NOT NULL,
  cost_assumptions_json JSONB NOT NULL,
  decision TEXT NOT NULL,
  decision_reason TEXT NOT NULL,
  system_mode TEXT NOT NULL,
  code_version TEXT NOT NULL,
  config_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_runs (
  run_id TEXT PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL,
  config_json JSONB NOT NULL,
  metrics_json JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_trades (
  trade_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES backtest_runs(run_id),
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  entry_time TIMESTAMPTZ NOT NULL,
  exit_time TIMESTAMPTZ,
  pnl NUMERIC NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_orders (
  order_id TEXT PRIMARY KEY,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  quantity NUMERIC NOT NULL,
  status TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_fills (
  fill_id TEXT PRIMARY KEY,
  order_id TEXT NOT NULL REFERENCES paper_orders(order_id),
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  quantity NUMERIC NOT NULL,
  price NUMERIC NOT NULL,
  fee NUMERIC NOT NULL,
  filled_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_intents (
  intent_id TEXT PRIMARY KEY,
  model_run_id TEXT NOT NULL,
  signal_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  event_time TIMESTAMPTZ NOT NULL,
  status TEXT NOT NULL,
  approved_for_paper_demo BOOLEAN NOT NULL,
  created_at TIMESTAMPTZ NOT NULL,
  intent_json JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS paper_intents_model_created_idx
  ON paper_intents (model_run_id, created_at DESC);

CREATE INDEX IF NOT EXISTS paper_intents_symbol_created_idx
  ON paper_intents (symbol, created_at DESC);

CREATE INDEX IF NOT EXISTS paper_intents_status_created_idx
  ON paper_intents (status, created_at DESC);

CREATE TABLE IF NOT EXISTS paper_intent_executions (
  intent_id TEXT PRIMARY KEY REFERENCES paper_intents(intent_id),
  order_id TEXT NOT NULL REFERENCES paper_orders(order_id),
  fill_id TEXT NOT NULL REFERENCES paper_fills(fill_id),
  created_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS paper_intent_executions_created_idx
  ON paper_intent_executions (created_at DESC);

CREATE TABLE IF NOT EXISTS paper_trade_closes (
  close_id TEXT PRIMARY KEY,
  entry_fill_id TEXT NOT NULL UNIQUE REFERENCES paper_fills(fill_id),
  order_id TEXT NOT NULL REFERENCES paper_orders(order_id),
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  quantity NUMERIC NOT NULL,
  entry_price NUMERIC NOT NULL,
  exit_price NUMERIC NOT NULL,
  gross_pnl NUMERIC NOT NULL,
  total_fees_paid NUMERIC NOT NULL,
  realized_pnl NUMERIC NOT NULL,
  close_reason TEXT NOT NULL,
  closed_at TIMESTAMPTZ NOT NULL,
  close_json JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS paper_trade_closes_entry_fill_idx
  ON paper_trade_closes (entry_fill_id);

CREATE INDEX IF NOT EXISTS paper_trade_closes_symbol_closed_idx
  ON paper_trade_closes (symbol, closed_at DESC);

CREATE INDEX IF NOT EXISTS paper_trade_closes_closed_idx
  ON paper_trade_closes (closed_at DESC);

CREATE TABLE IF NOT EXISTS paper_lifecycle_events (
  event_id TEXT PRIMARY KEY,
  event_type TEXT NOT NULL,
  entity_type TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  related_entity_id TEXT,
  created_at TIMESTAMPTZ NOT NULL,
  event_json JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS paper_lifecycle_events_entity_idx
  ON paper_lifecycle_events (entity_type, entity_id, created_at DESC);

CREATE INDEX IF NOT EXISTS paper_lifecycle_events_type_idx
  ON paper_lifecycle_events (event_type, created_at DESC);

CREATE INDEX IF NOT EXISTS paper_lifecycle_events_created_idx
  ON paper_lifecycle_events (created_at DESC);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
  snapshot_id TEXT PRIMARY KEY,
  event_time TIMESTAMPTZ NOT NULL,
  equity NUMERIC NOT NULL,
  margin_used NUMERIC NOT NULL,
  available_margin NUMERIC NOT NULL,
  positions_json JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS risk_events (
  risk_event_id TEXT PRIMARY KEY,
  event_time TIMESTAMPTZ NOT NULL,
  symbol TEXT,
  severity TEXT NOT NULL,
  reason TEXT NOT NULL,
  details_json JSONB NOT NULL
);
