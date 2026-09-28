"""
Thin wrapper around the official `dhanhq` SDK.

Every other module talks to Dhan only through this file — if the SDK's
interface changes, or you move to raw REST calls, only this file needs to
change.

Reference: DhanHQ-py README (https://github.com/dhan-oss/DhanHQ-py) and
DhanHQ v2 API docs (https://dhanhq.co/docs/v2/historical-data/).

Rate limit: Dhan's published non-trading API limit is 20 requests/second
(https://dhan.freshdesk.com/support/solutions/articles/82000891163). We
default to a conservative 5/sec via config.dhan.requests_per_second — raise
it if you've confirmed your plan's actual limit.

Note on delivery %: Dhan's historical daily-candle endpoint returns only
open/high/low/close/volume/timestamp — no delivery quantity. Delivery %
must come from NSE's daily bhavcopy / security-wise delivery position
files. See `ingest_ohlcv.py` for how the two sources are merged.
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime
from typing import Optional

import pandas as pd

from config import CONFIG

logger = logging.getLogger(__name__)

try:
    from dhanhq import DhanContext, dhanhq
except ImportError:  # pragma: no cover - surfaced clearly at call time instead
    DhanContext = None
    dhanhq = None


class DhanClient:
    """Wraps dhanhq for historical OHLCV pulls with basic rate limiting."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        access_token: Optional[str] = None,
    ) -> None:
        if dhanhq is None:
            raise ImportError(
                "The 'dhanhq' package is not installed. Run "
                "`pip install -r requirements.txt` first."
            )

        self.client_id = client_id or CONFIG.dhan.client_id
        self.access_token = access_token or CONFIG.dhan.access_token

        if not self.client_id or not self.access_token:
            raise ValueError(
                "DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN are not set. Copy "
                ".env.example to .env and fill them in."
            )

        context = DhanContext(self.client_id, self.access_token)
        self._dhan = dhanhq(context)

        self._min_interval = 1.0 / max(CONFIG.dhan.requests_per_second, 0.1)
        self._last_call_ts = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_ts = time.monotonic()

    def get_historical_daily(
        self,
        security_id: str,
        symbol: str,
        exchange_segment: str = "NSE_EQ",
        instrument: str = "EQUITY",
        from_date: date | str = None,
        to_date: date | str = None,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV candles for one symbol.

        Returns a DataFrame with columns: date, symbol, open, high, low,
        close, volume. Delivery columns are NOT populated here — see
        ingest_ohlcv.py for merging in NSE delivery data.
        """
        from_date_str = self._to_date_str(from_date)
        to_date_str = self._to_date_str(to_date)

        self._throttle()
        response = self._dhan.historical_daily_data(
            security_id=security_id,
            exchange_segment=exchange_segment,
            instrument_type=instrument,
            from_date=from_date_str,
            to_date=to_date_str,
        )

        return self._parse_candle_response(response, symbol)

    @staticmethod
    def _to_date_str(value: date | str | None) -> str:
        if value is None:
            raise ValueError("from_date/to_date must be provided")
        if isinstance(value, str):
            return value
        return value.strftime("%Y-%m-%d")

    @staticmethod
    def _parse_candle_response(response: dict, symbol: str) -> pd.DataFrame:
        """Parse the dhanhq historical_daily_data response into a tidy df.

        The SDK returns either the raw payload directly (parallel arrays:
        open/high/low/close/volume/timestamp) or a wrapper
        {"status": ..., "data": {...}} depending on SDK version/error
        state. Handle both.
        """
        if response is None:
            logger.warning("Empty response for %s", symbol)
            return pd.DataFrame()

        if "data" in response and isinstance(response["data"], dict):
            payload = response["data"]
        else:
            payload = response

        if response.get("status") == "failure":
            remarks = response.get("remarks", {})
            logger.error(
                "Dhan API error for %s: %s", symbol, remarks.get("error_message", remarks)
            )
            return pd.DataFrame()

        required = ("open", "high", "low", "close", "volume", "timestamp")
        if not all(k in payload for k in required):
            logger.error(
                "Unexpected response shape for %s, missing keys. Got: %s",
                symbol, list(payload.keys()),
            )
            return pd.DataFrame()

        df = pd.DataFrame({
            "date": [datetime.fromtimestamp(ts).date() for ts in payload["timestamp"]],
            "symbol": symbol,
            "open": payload["open"],
            "high": payload["high"],
            "low": payload["low"],
            "close": payload["close"],
            "volume": payload["volume"],
        })
        df["date"] = pd.to_datetime(df["date"])
        return df


_client_singleton: Optional[DhanClient] = None


def get_dhan_client() -> DhanClient:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = DhanClient()
    return _client_singleton
