from __future__ import annotations

from app.ingestion import scheduler


def test_scheduler_default_mode_routes_to_public_backfill(monkeypatch) -> None:
    called: dict[str, list[str]] = {}

    def _fake_main(args: list[str]) -> None:
        called["args"] = list(args)

    monkeypatch.setattr(scheduler, "public_backfill_main", _fake_main)
    monkeypatch.setattr("sys.argv", ["scheduler"])

    scheduler.main()

    assert "args" in called


def test_scheduler_backfill_alias_kept_for_cli_compatibility(monkeypatch) -> None:
    called: dict[str, list[str]] = {}

    def _fake_main(args: list[str]) -> None:
        called["args"] = list(args)

    monkeypatch.setattr(scheduler, "public_backfill_main", _fake_main)
    monkeypatch.setattr("sys.argv", ["scheduler", "--mode", "backfill", "--symbol", "BTCUSDT"])

    scheduler.main()

    assert called["args"] == ["--symbol", "BTCUSDT"]
