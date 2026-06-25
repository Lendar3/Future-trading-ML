from __future__ import annotations

import os

import pytest

from app.core.safety import LiveTradingDisabledError
from app.execution.real_disabled import RealBrokerInterface
from app.ingestion.bybit_rest import BybitPublicRestClient


def test_bybit_public_client_requires_no_private_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BYBIT_API_KEY", raising=False)
    monkeypatch.delenv("BYBIT_API_SECRET", raising=False)
    client = BybitPublicRestClient()
    assert "X-BAPI-API-KEY" not in client.default_headers
    assert os.getenv("BYBIT_API_KEY") is None


def test_live_trading_remains_disabled_with_public_ingestion_present() -> None:
    with pytest.raises(LiveTradingDisabledError):
        RealBrokerInterface().place_order(symbol="BTCUSDT", side="buy", quantity=1)
