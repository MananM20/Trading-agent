"""
Phase 2: Daily shortlist runner.

Runs both screens against every symbol in the universe for the most recent
trading day in storage, and writes a shortlist file via
DataProvider.save_shortlist(): date, symbol, screen_name, reason_tags,
entry_price, stop_loss, target_or_trailing_flag.

"Flagged" semantics (Phase 0 decision, logged here to avoid double-counting
in backtests): a symbol is only written to the shortlist on the day its
screen condition FIRST fires. If it fired yesterday too, it's not
re-flagged today even if still passing.

Usage:
    python -m signals.run_screens
    python -m signals.run_screens --date 2025-06-10
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from config import CONFIG
from data_pipeline.storage import get_data_provider
from signals.beaten_down_screen import evaluate_beaten_down
from signals.breakout_screen import evaluate_breakout

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NIFTY50_SYMBOL = "NIFTY50"  # expected to exist in OHLCV store as an index series


def _first_fire_only(df: pd.DataFrame, screen_name: str, run_date: pd.Timestamp) -> pd.DataFrame:
    """Drop rows for symbols whose screen condition was already true on the
    prior trading day (i.e., keep only newly-firing flags)."""
    provider = get_data_provider()
    prev_shortlist = provider.get_shortlist(run_date - pd.Timedelta(days=1))
    if prev_shortlist.empty:
        return df
    already_flagged = set(
        prev_shortlist.loc[prev_shortlist["screen_name"] == screen_name, "symbol"]
    )
    return df[~df["symbol"].isin(already_flagged)]


def run_daily_screens(run_date: pd.Timestamp | None = None) -> pd.DataFrame:
    provider = get_data_provider()
    universe = provider.get_universe()
    peer_groups = provider.get_peer_groups().set_index("symbol")["peers"].to_dict()

    nifty_df = provider.get_ohlcv(NIFTY50_SYMBOL)
    if nifty_df.empty:
        logger.warning(
            "No Nifty 50 OHLCV found under symbol '%s' — breakout screen's "
            "relative-strength rule (#5) will fail for every symbol until "
            "this index series is ingested.", NIFTY50_SYMBOL,
        )

    rows = []
    price_history_cache: dict[str, pd.DataFrame] = {}

    for symbol in universe["symbol"]:
        df = provider.get_ohlcv(symbol)
        if df.empty:
            continue
        price_history_cache[symbol] = df

        if run_date is not None:
            df = df[df["date"] <= run_date]
        if df.empty:
            continue

        actual_date = df["date"].iloc[-1]

        breakout = evaluate_breakout(symbol, df, nifty_df)
        if breakout.passed:
            rows.append({
                "date": actual_date,
                "symbol": symbol,
                "screen_name": CONFIG.screens.breakout,
                "reason_tags": ",".join(breakout.reason_tags),
                "entry_price": breakout.entry_price,
                "stop_loss": breakout.stop_loss,
                "target_or_trailing_flag": breakout.target_or_trailing,
            })

    for symbol in universe["symbol"]:
        df = price_history_cache.get(symbol)
        if df is None or df.empty:
            continue
        if run_date is not None:
            df = df[df["date"] <= run_date]
        if df.empty:
            continue

        actual_date = df["date"].iloc[-1]
        fundamentals = provider.get_fundamentals(symbol)
        peers_raw = peer_groups.get(symbol, "")
        peer_symbols = [p for p in str(peers_raw).split("|") if p] if pd.notna(peers_raw) else []
        for peer in peer_symbols:
            if peer not in price_history_cache:
                price_history_cache[peer] = provider.get_ohlcv(peer)

        beaten_down = evaluate_beaten_down(
            symbol, df, fundamentals, peer_symbols, price_history_cache
        )
        if beaten_down.passed:
            rows.append({
                "date": actual_date,
                "symbol": symbol,
                "screen_name": CONFIG.screens.beaten_down,
                "reason_tags": ",".join(beaten_down.reason_tags),
                "entry_price": beaten_down.entry_price,
                "stop_loss": beaten_down.stop_loss,
                "target_or_trailing_flag": beaten_down.target_or_trailing,
            })

    result = pd.DataFrame(rows, columns=[
        "date", "symbol", "screen_name", "reason_tags",
        "entry_price", "stop_loss", "target_or_trailing_flag",
    ])

    if result.empty:
        logger.info("No symbols passed either screen for this run.")
        return result

    effective_date = pd.to_datetime(result["date"].max())
    for screen_name in result["screen_name"].unique():
        mask = result["screen_name"] == screen_name
        result.loc[mask] = pd.concat([
            _first_fire_only(result[mask], screen_name, effective_date),
        ])

    result = result.dropna(subset=["symbol"])
    provider.save_shortlist(effective_date, result)
    logger.info("Saved shortlist for %s: %d symbols flagged", effective_date.date(), len(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 daily screens")
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD, defaults to latest available")
    args = parser.parse_args()

    run_date = pd.to_datetime(args.date) if args.date else None
    run_daily_screens(run_date)


if __name__ == "__main__":
    main()
