"""
Phase 1: Pull OHLCV + delivery % for the full universe, run data quality
checks, and store via the DataProvider.

Two data sources are combined:
  - Dhan historical daily candle API -> open/high/low/close/volume
  - NSE bhavcopy (via jugaad-data)   -> delivery_qty/delivery_pct
    (Dhan's historical endpoint does not return delivery data)

Usage:
    python -m data_pipeline.ingest_ohlcv --years 2
    python -m data_pipeline.ingest_ohlcv --years 2 --symbols RELIANCE,TCS
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta

import pandas as pd

from config import CONFIG
from data_pipeline.dhan_client import get_dhan_client
from data_pipeline.instrument_master import build_symbol_to_security_id_map
from data_pipeline.quality_checks import run_quality_checks
from data_pipeline.storage import get_data_provider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def fetch_delivery_data(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Fetch delivery_qty / delivery_pct for a symbol from NSE bhavcopy via
    jugaad-data. Returns empty df (with the right columns) on failure —
    ingestion should not hard-fail just because delivery data is
    unavailable for a given stretch of dates (NSE's format changed on
    2024-07-08, and pre-that-date requests use a different code path)."""
    try:
        from jugaad_data.nse import stock_df
    except ImportError:
        logger.warning(
            "jugaad-data not installed — delivery_pct will be left null. "
            "Run `pip install jugaad-data` to enable it."
        )
        return pd.DataFrame(columns=["date", "delivery_qty", "delivery_pct"])

    try:
        df = stock_df(symbol=symbol, from_date=start, to_date=end, series="EQ")
    except Exception as exc:  # NSE scraping is inherently flaky
        logger.warning("Delivery data fetch failed for %s: %s", symbol, exc)
        return pd.DataFrame(columns=["date", "delivery_qty", "delivery_pct"])

    if df.empty:
        return pd.DataFrame(columns=["date", "delivery_qty", "delivery_pct"])

    df = df.rename(columns={
        "DATE": "date",
        "DELIV_QTY": "delivery_qty",
        "DELIV_PER": "delivery_pct",
    })
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "delivery_qty", "delivery_pct"]]


def ingest_symbol(
    symbol: str,
    security_id: str,
    start: date,
    end: date,
    holidays: set,
) -> dict:
    provider = get_data_provider()
    client = get_dhan_client()

    ohlcv = client.get_historical_daily(
        security_id=security_id,
        symbol=symbol,
        from_date=start,
        to_date=end,
    )
    if ohlcv.empty:
        logger.warning("No OHLCV data returned for %s", symbol)
        return {"symbol": symbol, "passed": False, "notes": "no data returned"}

    delivery = fetch_delivery_data(symbol, start, end)
    if not delivery.empty:
        ohlcv = ohlcv.merge(delivery, on="date", how="left")
    else:
        ohlcv["delivery_qty"] = pd.NA
        ohlcv["delivery_pct"] = pd.NA

    report = run_quality_checks(symbol, ohlcv, holidays)

    provider.save_ohlcv(ohlcv)
    logger.info("Ingested %d rows for %s (quality passed=%s)", len(ohlcv), symbol, report.passed)

    return report.as_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest OHLCV + delivery data")
    parser.add_argument("--years", type=int, default=CONFIG.backtest.lookback_years)
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="Comma-separated symbols to override the universe (for testing)"
    )
    args = parser.parse_args()

    provider = get_data_provider()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        universe = provider.get_universe()
        symbols = universe["symbol"].tolist()

    end = date.today()
    start = end - timedelta(days=365 * args.years)
    holidays = provider.get_holiday_calendar()

    if not holidays:
        logger.warning(
            "No holiday calendar loaded. Run "
            "`python -m data_pipeline.ingest_holidays` first for accurate "
            "missing-day checks."
        )

    logger.info("Resolving %d symbols to Dhan security IDs...", len(symbols))
    security_id_map = build_symbol_to_security_id_map()

    reports = []
    for symbol in symbols:
        security_id = security_id_map.get(symbol)
        if security_id is None:
            logger.error("No security_id found for %s — skipping", symbol)
            reports.append({"symbol": symbol, "passed": False, "notes": "security_id not found"})
            continue

        try:
            report = ingest_symbol(symbol, security_id, start, end, holidays)
        except Exception as exc:
            logger.exception("Ingestion failed for %s", symbol)
            report = {"symbol": symbol, "passed": False, "notes": str(exc)}
        reports.append(report)

    summary = pd.DataFrame(reports)
    summary_path = CONFIG.storage.ohlcv_dir.parent / "ingestion_report.csv"
    summary.to_csv(summary_path, index=False)

    n_passed = summary["passed"].sum() if "passed" in summary else 0
    logger.info(
        "Ingestion complete: %d/%d symbols passed quality checks. Report: %s",
        n_passed, len(symbols), summary_path,
    )


if __name__ == "__main__":
    main()
