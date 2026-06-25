from __future__ import annotations


class LiveTradingDisabledError(RuntimeError):
    """Raised whenever v1 code attempts real live execution."""


def raise_live_trading_disabled() -> None:
    raise LiveTradingDisabledError(
        "Live trading is disabled in v1. Use research, backtest, paper, demo, "
        "or real_advisory modes only."
    )


def assert_real_trading_disabled(real_trading_enabled: bool) -> None:
    if real_trading_enabled:
        raise LiveTradingDisabledError("Configuration attempted to enable real trading in v1.")
