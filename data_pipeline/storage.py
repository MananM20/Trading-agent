"""
DataProvider: the single interface every other phase uses to read/write
market data, fundamentals, and the NSE holiday calendar.

Storage backend is Parquet-on-disk by default (config.storage.backend ==
"parquet"), laid out as data/eod/{symbol}/{year}.parquet per the plan's
suggested structure. Signal/screen/backtest code should never touch pandas
I/O directly for these datasets — go through DataProvider so the backend
(e.g. swapping to Postgres later) can change without touching callers.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from config import CONFIG

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = [
    "date", "symbol", "open", "high", "low", "close", "volume",
    "delivery_qty", "delivery_pct",
]

FUNDAMENTALS_COLUMNS = [
    "symbol", "quarter", "revenue", "pat", "roe", "roce",
    "debt_to_equity", "promoter_holding", "pledged_pct",
    "operating_cash_flow",
]


class DataProvider:
    """Parquet-backed implementation of the storage interface.

    Layout:
      data/eod/{symbol}/{year}.parquet         — one file per symbol per year
      data/fundamentals/{symbol}.parquet       — one file per symbol, all quarters
      data/nse_holidays.csv                    — flat holiday calendar
      data/shortlists/{date}.csv               — daily screen output
    """

    def __init__(self) -> None:
        self.ohlcv_dir: Path = CONFIG.storage.ohlcv_dir
        self.fundamentals_dir: Path = CONFIG.storage.fundamentals_dir
        self.shortlist_dir: Path = CONFIG.storage.shortlist_dir
        self.holiday_file: Path = CONFIG.storage.holiday_calendar_file

        self.ohlcv_dir.mkdir(parents=True, exist_ok=True)
        self.fundamentals_dir.mkdir(parents=True, exist_ok=True)
        self.shortlist_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------
    def save_ohlcv(self, df: pd.DataFrame) -> None:
        """Persist OHLCV rows, partitioned by symbol and year.

        Expects columns matching OHLCV_COLUMNS (extra columns are kept).
        Existing rows for the same (symbol, date) are overwritten, not
        duplicated.
        """
        if df.empty:
            return
        missing = set(["date", "symbol"]) - set(df.columns)
        if missing:
            raise ValueError(f"save_ohlcv missing required columns: {missing}")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["_year"] = df["date"].dt.year

        for (symbol, year), group in df.groupby(["symbol", "_year"]):
            symbol_dir = self.ohlcv_dir / str(symbol)
            symbol_dir.mkdir(parents=True, exist_ok=True)
            path = symbol_dir / f"{year}.parquet"
            group = group.drop(columns=["_year"])

            if path.exists():
                existing = pd.read_parquet(path)
                existing["date"] = pd.to_datetime(existing["date"])
                combined = pd.concat([existing, group], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=["symbol", "date"], keep="last"
                )
            else:
                combined = group

            combined = combined.sort_values("date").reset_index(drop=True)
            combined.to_parquet(path, index=False)

    def get_ohlcv(
        self,
        symbol: str,
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pd.DataFrame:
        """Return OHLCV rows for a symbol between start and end (inclusive)."""
        symbol_dir = self.ohlcv_dir / symbol
        if not symbol_dir.exists():
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        frames = []
        for path in sorted(symbol_dir.glob("*.parquet")):
            frames.append(pd.read_parquet(path))

        if not frames:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])

        if start is not None:
            df = df[df["date"] >= pd.to_datetime(start)]
        if end is not None:
            df = df[df["date"] <= pd.to_datetime(end)]

        return df.sort_values("date").reset_index(drop=True)

    def get_ohlcv_multi(
        self,
        symbols: list[str],
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pd.DataFrame:
        """Convenience: concat get_ohlcv across many symbols."""
        frames = [self.get_ohlcv(sym, start, end) for sym in symbols]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Fundamentals
    # ------------------------------------------------------------------
    def save_fundamentals(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        missing = {"symbol", "quarter"} - set(df.columns)
        if missing:
            raise ValueError(f"save_fundamentals missing required columns: {missing}")

        for symbol, group in df.groupby("symbol"):
            path = self.fundamentals_dir / f"{symbol}.parquet"
            if path.exists():
                existing = pd.read_parquet(path)
                combined = pd.concat([existing, group], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=["symbol", "quarter"], keep="last"
                )
            else:
                combined = group
            combined.to_parquet(path, index=False)

    def get_fundamentals(self, symbol: str) -> pd.DataFrame:
        path = self.fundamentals_dir / f"{symbol}.parquet"
        if not path.exists():
            return pd.DataFrame(columns=FUNDAMENTALS_COLUMNS)
        return pd.read_parquet(path)

    # ------------------------------------------------------------------
    # NSE holiday calendar
    # ------------------------------------------------------------------
    def save_holiday_calendar(self, holidays: list[date]) -> None:
        df = pd.DataFrame({"date": pd.to_datetime(holidays)})
        df = df.drop_duplicates().sort_values("date")
        df.to_csv(self.holiday_file, index=False)

    def get_holiday_calendar(self) -> set[pd.Timestamp]:
        if not self.holiday_file.exists():
            logger.warning(
                "No NSE holiday calendar found at %s — missing-day gap "
                "checks will treat every weekday as a trading day.",
                self.holiday_file,
            )
            return set()
        df = pd.read_csv(self.holiday_file, parse_dates=["date"])
        return set(df["date"])

    # ------------------------------------------------------------------
    # Shortlists (Phase 2 output)
    # ------------------------------------------------------------------
    def save_shortlist(self, run_date: date | str, df: pd.DataFrame) -> Path:
        run_date_str = pd.to_datetime(run_date).strftime("%Y-%m-%d")
        path = self.shortlist_dir / f"{run_date_str}.csv"
        df.to_csv(path, index=False)
        return path

    def get_shortlist(self, run_date: date | str) -> pd.DataFrame:
        run_date_str = pd.to_datetime(run_date).strftime("%Y-%m-%d")
        path = self.shortlist_dir / f"{run_date_str}.csv"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, parse_dates=["date"])

    # ------------------------------------------------------------------
    # Universe / peer groups (static config-driven CSVs)
    # ------------------------------------------------------------------
    def get_universe(self) -> pd.DataFrame:
        path = CONFIG.universe.universe_file
        if not path.exists():
            raise FileNotFoundError(
                f"Universe file not found at {path}. See config.py "
                f"UniverseConfig.universe_file."
            )
        return pd.read_csv(path)

    def get_peer_groups(self) -> pd.DataFrame:
        path = CONFIG.universe.peer_group_file
        if not path.exists():
            return pd.DataFrame(columns=["symbol", "sector", "peers"])
        return pd.read_csv(path)


def get_data_provider() -> DataProvider:
    """Factory — swap backend here based on CONFIG.storage.backend."""
    if CONFIG.storage.backend == "parquet":
        return DataProvider()
    raise NotImplementedError(
        f"Storage backend '{CONFIG.storage.backend}' is not implemented yet. "
        f"Only 'parquet' is supported currently."
    )
