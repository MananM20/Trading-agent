"""
Automated data quality checks for OHLCV data (Phase 1 acceptance criteria).

These are meant to be run after every ingestion, not as manual spot checks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from config import CONFIG

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    symbol: str
    total_rows: int = 0
    duplicate_rows: int = 0
    missing_trading_days: int = 0
    missing_day_pct: float = 0.0
    corporate_action_suspects: list = field(default_factory=list)
    delivery_pct_out_of_range: int = 0
    passed: bool = True
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "total_rows": self.total_rows,
            "duplicate_rows": self.duplicate_rows,
            "missing_trading_days": self.missing_trading_days,
            "missing_day_pct": round(self.missing_day_pct, 3),
            "corporate_action_suspects": len(self.corporate_action_suspects),
            "delivery_pct_out_of_range": self.delivery_pct_out_of_range,
            "passed": self.passed,
            "notes": "; ".join(self.notes),
        }


def check_duplicates(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    dup_mask = df.duplicated(subset=["symbol", "date"], keep=False)
    return int(dup_mask.sum())


def check_missing_trading_days(
    df: pd.DataFrame, holidays: set[pd.Timestamp]
) -> tuple[int, float]:
    """Compare actual trading days present against the expected calendar
    (all weekdays between min/max date, minus NSE holidays)."""
    if df.empty:
        return 0, 0.0

    start, end = df["date"].min(), df["date"].max()
    all_days = pd.bdate_range(start, end)  # business days (Mon-Fri)
    expected_days = pd.DatetimeIndex([d for d in all_days if d not in holidays])

    present_days = pd.DatetimeIndex(df["date"].unique())
    missing = expected_days.difference(present_days)

    missing_pct = (len(missing) / len(expected_days) * 100) if len(expected_days) else 0.0
    return len(missing), missing_pct


def check_corporate_action_suspects(df: pd.DataFrame) -> list[dict]:
    """Flag rows with a >threshold single-day price change alongside a
    volume anomaly (likely unadjusted split/bonus), for manual review."""
    if df.empty or len(df) < 2:
        return []

    df = df.sort_values("date").copy()
    df["pct_change"] = df["close"].pct_change() * 100
    avg_volume = df["volume"].rolling(20, min_periods=5).mean()
    df["volume_ratio"] = df["volume"] / avg_volume

    threshold = CONFIG.data_quality.max_single_day_move_pct
    suspects = df[
        (df["pct_change"].abs() > threshold) & (df["volume_ratio"] > 1.5)
    ]

    return suspects[["date", "close", "pct_change", "volume", "volume_ratio"]].to_dict(
        "records"
    )


def check_delivery_pct_range(df: pd.DataFrame) -> int:
    if "delivery_pct" not in df.columns or df.empty:
        return 0
    lo = CONFIG.data_quality.delivery_pct_min
    hi = CONFIG.data_quality.delivery_pct_max
    out_of_range = df[
        df["delivery_pct"].notna()
        & ((df["delivery_pct"] < lo) | (df["delivery_pct"] > hi))
    ]
    return len(out_of_range)


def run_quality_checks(
    symbol: str, df: pd.DataFrame, holidays: set[pd.Timestamp]
) -> QualityReport:
    report = QualityReport(symbol=symbol, total_rows=len(df))

    if df.empty:
        report.passed = False
        report.notes.append("No data ingested")
        return report

    report.duplicate_rows = check_duplicates(df)
    if report.duplicate_rows > 0:
        report.passed = False
        report.notes.append(f"{report.duplicate_rows} duplicate (symbol, date) rows")

    missing_days, missing_pct = check_missing_trading_days(df, holidays)
    report.missing_trading_days = missing_days
    report.missing_day_pct = missing_pct
    if missing_pct > CONFIG.data_quality.max_missing_day_pct:
        report.passed = False
        report.notes.append(
            f"{missing_pct:.2f}% missing trading days exceeds "
            f"{CONFIG.data_quality.max_missing_day_pct}% threshold"
        )

    report.corporate_action_suspects = check_corporate_action_suspects(df)
    if report.corporate_action_suspects:
        report.notes.append(
            f"{len(report.corporate_action_suspects)} rows flagged for possible "
            f"unadjusted corporate action — review manually"
        )

    report.delivery_pct_out_of_range = check_delivery_pct_range(df)
    if report.delivery_pct_out_of_range > 0:
        report.passed = False
        report.notes.append(
            f"{report.delivery_pct_out_of_range} rows have delivery_pct outside [0, 100]"
        )

    return report
