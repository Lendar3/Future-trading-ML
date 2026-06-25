from __future__ import annotations

import json
import ssl
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.core.types import RawBar

BYBIT_PUBLIC_BASE_URL = "https://api.bybit.com"
DEFAULT_CATEGORY = "linear"


class BybitPublicRestError(RuntimeError):
    pass


class BybitRateLimitError(BybitPublicRestError):
    pass


def utc_from_ms(value: int | str) -> datetime:
    return datetime.fromtimestamp(int(value) / 1000, tz=UTC)


def utc_ms(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return int(value.astimezone(UTC).timestamp() * 1000)


def interval_to_milliseconds(interval: str) -> int:
    if interval.isdigit():
        return int(interval) * 60_000
    mapping = {
        "D": 24 * 60 * 60_000,
        "W": 7 * 24 * 60 * 60_000,
        "M": 30 * 24 * 60 * 60_000,
    }
    if interval not in mapping:
        raise ValueError(f"Unsupported Bybit interval: {interval}")
    return mapping[interval]


def iter_time_chunks(
    start: datetime,
    end: datetime,
    interval: str,
    limit: int,
) -> list[tuple[datetime, datetime]]:
    if end < start:
        raise ValueError("end must be greater than or equal to start")
    step_ms = interval_to_milliseconds(interval) * limit
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(milliseconds=step_ms - 1))
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(milliseconds=1)
    return chunks


def parse_kline_row(
    row: list[str],
    symbol: str,
    timeframe: str,
    fetch_time: datetime,
    source: str,
) -> RawBar:
    return RawBar(
        symbol=symbol,
        timeframe=timeframe,
        event_time=utc_from_ms(row[0]),
        fetch_time=fetch_time,
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]) if len(row) > 5 else 0.0,
        source=source,
        metadata={"turnover": float(row[6]) if len(row) > 6 else None},
    )


def raw_bar_to_upsert_row(bar: RawBar) -> dict[str, Any]:
    return {
        "symbol": bar.symbol,
        "timeframe": bar.timeframe,
        "event_time": bar.event_time,
        "fetch_time": bar.fetch_time,
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
        "source": bar.source,
        "metadata_json": json.dumps(bar.metadata),
    }


def parse_funding_item(item: dict[str, Any], fetch_time: datetime, source: str) -> dict[str, Any]:
    return {
        "symbol": str(item["symbol"]),
        "event_time": utc_from_ms(item["fundingRateTimestamp"]),
        "fetch_time": fetch_time,
        "funding_rate": float(item["fundingRate"]),
        "source": source,
        "metadata_json": json.dumps(item),
    }


def parse_open_interest_item(
    item: dict[str, Any],
    symbol: str,
    fetch_time: datetime,
    source: str,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "event_time": utc_from_ms(item["timestamp"]),
        "fetch_time": fetch_time,
        "open_interest": float(item["openInterest"]),
        "source": source,
        "metadata_json": json.dumps(item),
    }


def parse_long_short_ratio_item(
    item: dict[str, Any],
    symbol: str,
    fetch_time: datetime,
    source: str,
) -> dict[str, Any]:
    buy_ratio = float(item.get("buyRatio", 0.0))
    sell_ratio = float(item.get("sellRatio", 0.0))
    ratio = buy_ratio / sell_ratio if sell_ratio else 0.0
    return {
        "symbol": symbol,
        "event_time": utc_from_ms(item["timestamp"]),
        "fetch_time": fetch_time,
        "long_short_ratio": ratio,
        "source": source,
        "metadata_json": json.dumps(item),
    }


@dataclass(frozen=True)
class BybitPublicRestClient:
    base_url: str = BYBIT_PUBLIC_BASE_URL
    category: str = DEFAULT_CATEGORY
    timeout_sec: int = 15
    max_retries: int = 3
    backoff_sec: float = 0.5
    source: str = "bybit_public"
    default_headers: dict[str, str] = field(
        default_factory=lambda: {"User-Agent": "BybitFuturesEdgeLab/0.1"}
    )

    def _ssl_context(self) -> ssl.SSLContext | None:
        try:
            import certifi
        except ImportError:
            return None
        return ssl.create_default_context(cafile=certifi.where())

    def _request(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        query = urlencode({key: value for key, value in params.items() if value is not None})
        url = f"{self.base_url}{path}?{query}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            request = Request(url, headers=self.default_headers, method="GET")
            try:
                with urlopen(
                    request,
                    timeout=self.timeout_sec,
                    context=self._ssl_context(),
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                last_error = exc
                if exc.code in {429, 403} and attempt < self.max_retries:
                    time.sleep(self.backoff_sec * (2**attempt))
                    continue
                raise BybitPublicRestError(f"Bybit HTTP error {exc.code} for {path}") from exc
            except URLError as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_sec * (2**attempt))
                    continue
                raise BybitPublicRestError(f"Bybit network error for {path}: {exc}") from exc

            ret_code = int(payload.get("retCode", -1))
            if ret_code == 0:
                return cast(dict[str, Any], payload)
            if ret_code == 10006 and attempt < self.max_retries:
                time.sleep(self.backoff_sec * (2**attempt))
                continue
            if ret_code == 10006:
                raise BybitRateLimitError(str(payload))
            raise BybitPublicRestError(str(payload))

        raise BybitPublicRestError(f"Bybit request failed for {path}: {last_error}")

    def _fetch_kline_endpoint(
        self,
        endpoint: str,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int = 1000,
    ) -> list[RawBar]:
        rows: list[RawBar] = []
        for chunk_start, chunk_end in iter_time_chunks(start, end, timeframe, limit):
            payload = self._request(
                endpoint,
                {
                    "category": self.category,
                    "symbol": symbol,
                    "interval": timeframe,
                    "start": utc_ms(chunk_start),
                    "end": utc_ms(chunk_end),
                    "limit": limit,
                },
            )
            fetch_time = datetime.fromtimestamp(payload["time"] / 1000, tz=UTC)
            raw_rows = payload.get("result", {}).get("list", [])
            rows.extend(
                parse_kline_row(
                    row=item,
                    symbol=symbol,
                    timeframe=timeframe,
                    fetch_time=fetch_time,
                    source=self.source,
                )
                for item in raw_rows
            )
        return sorted(rows, key=lambda item: item.event_time)

    def fetch_klines(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int = 1000,
    ) -> list[RawBar]:
        return self._fetch_kline_endpoint("/v5/market/kline", symbol, timeframe, start, end, limit)

    def fetch_mark_price_klines(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int = 1000,
    ) -> list[RawBar]:
        return self._fetch_kline_endpoint(
            "/v5/market/mark-price-kline", symbol, timeframe, start, end, limit
        )

    def fetch_index_price_klines(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int = 1000,
    ) -> list[RawBar]:
        return self._fetch_kline_endpoint(
            "/v5/market/index-price-kline", symbol, timeframe, start, end, limit
        )

    def fetch_funding_history(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        payload = self._request(
            "/v5/market/funding/history",
            {
                "category": self.category,
                "symbol": symbol,
                "startTime": utc_ms(start),
                "endTime": utc_ms(end),
                "limit": limit,
            },
        )
        fetch_time = datetime.fromtimestamp(payload["time"] / 1000, tz=UTC)
        return [
            parse_funding_item(item=item, fetch_time=fetch_time, source=self.source)
            for item in payload.get("result", {}).get("list", [])
        ]

    def fetch_open_interest(
        self,
        symbol: str,
        interval_time: str,
        start: datetime,
        end: datetime,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            payload = self._request(
                "/v5/market/open-interest",
                {
                    "category": self.category,
                    "symbol": symbol,
                    "intervalTime": interval_time,
                    "startTime": utc_ms(start),
                    "endTime": utc_ms(end),
                    "limit": limit,
                    "cursor": cursor,
                },
            )
            fetch_time = datetime.fromtimestamp(payload["time"] / 1000, tz=UTC)
            result = payload.get("result", {})
            output.extend(
                parse_open_interest_item(
                    item=item,
                    symbol=symbol,
                    fetch_time=fetch_time,
                    source=self.source,
                )
                for item in result.get("list", [])
            )
            cursor = result.get("nextPageCursor")
            if not cursor:
                break
        return sorted(output, key=lambda item: item["event_time"])

    def fetch_long_short_ratio(
        self,
        symbol: str,
        period: str,
        start: datetime,
        end: datetime,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            payload = self._request(
                "/v5/market/account-ratio",
                {
                    "category": self.category,
                    "symbol": symbol,
                    "period": period,
                    "startTime": utc_ms(start),
                    "endTime": utc_ms(end),
                    "limit": limit,
                    "cursor": cursor,
                },
            )
            fetch_time = datetime.fromtimestamp(payload["time"] / 1000, tz=UTC)
            result = payload.get("result", {})
            output.extend(
                parse_long_short_ratio_item(
                    item=item,
                    symbol=symbol,
                    fetch_time=fetch_time,
                    source=self.source,
                )
                for item in result.get("list", [])
            )
            cursor = result.get("nextPageCursor")
            if not cursor:
                break
        return sorted(output, key=lambda item: item["event_time"])

    def fetch_instruments_info(self, symbol: str | None = None) -> dict[str, Any]:
        return self._request(
            "/v5/market/instruments-info",
            {"category": self.category, "symbol": symbol},
        )

    def fetch_tickers(self, symbol: str | None = None) -> dict[str, Any]:
        return self._request(
            "/v5/market/tickers",
            {"category": self.category, "symbol": symbol},
        )


BybitRestClient = BybitPublicRestClient
