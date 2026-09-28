"""
Phase 1: Quarterly fundamentals ingestion.

screener.in / Tickertape don't offer a bulk, unauthenticated "export all
stocks" endpoint — screener.in's per-stock Excel export requires a logged-in
session and gives you one wide sheet per stock (quarters as columns), and
Tickertape's is similar. Scraping either at scale is against their terms of
use, so this script does NOT scrape screener.in.

Instead, this expects a single tidy CSV that you build once per quarterly
refresh (by hand, via screener.in's exports pasted into a sheet, or via
Screener's paid "export as CSV" from a saved stock screen at
https://www.screener.in/screens/ — a screen result table already exports
in roughly this shape), with one row per (symbol, quarter):

    symbol, quarter, revenue, pat, roe, roce, debt_to_equity,
    promoter_holding, pledged_pct, operating_cash_flow

`quarter` should be a consistent label, e.g. "Q1FY25", so quarter-over-quarter
comparisons in the beaten-down screen (Phase 2) sort correctly — see
`quarter_sort_key` below.

Usage:
    python -m data_pipeline.ingest_fundamentals path/to/fundamentals_export.csv
"""
from __future__ import annotations

import argparse
import logging
import re

import pandas as pd

from data_pipeline.storage import FUNDAMENTALS_COLUMNS, get_data_provider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_QUARTER_RE = re.compile(r"Q(?P<q>[1-4])FY(?P<y>\d{2,4})", re.IGNORECASE)


def quarter_sort_key(quarter: str) -> tuple[int, int]:
    """Turn 'Q1FY25' into (2025, 1) for chronological sorting."""
    match = _QUARTER_RE.match(quarter.strip())
    if not match:
        return (0, 0)
    year = int(match.group("y"))
    if year < 100:
        year += 2000
    return (year, int(match.group("q")))


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = set(FUNDAMENTALS_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Fundamentals CSV is missing required columns: {missing_cols}. "
            f"Expected: {FUNDAMENTALS_COLUMNS}"
        )

    df = df[FUNDAMENTALS_COLUMNS].copy()
    df = df.dropna(subset=["symbol", "quarter"])
    df["symbol"] = df["symbol"].str.strip().str.upper()

    numeric_cols = [c for c in FUNDAMENTALS_COLUMNS if c not in ("symbol", "quarter")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.drop_duplicates(subset=["symbol", "quarter"], keep="last")
    if len(df) < before:
        logger.warning("Dropped %d duplicate (symbol, quarter) rows", before - len(df))

    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest quarterly fundamentals CSV")
    parser.add_argument("csv_path", type=str, help="Path to the fundamentals export CSV")
    args = parser.parse_args()

    raw = pd.read_csv(args.csv_path)
    clean = validate_and_clean(raw)

    provider = get_data_provider()
    provider.save_fundamentals(clean)

    logger.info(
        "Ingested %d fundamentals rows across %d symbols",
        len(clean), clean["symbol"].nunique(),
    )


if __name__ == "__main__":
    main()
