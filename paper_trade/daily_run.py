"""
Phase 4: Local paper trading — one EOD run per trading day.

Each invocation (intended to run once per day after market close, e.g. via
Windows Task Scheduler):
  1. Ingests the latest OHLCV for the universe (delegates to
     data_pipeline.ingest_ohlcv for a fresh pull, or you can run that
     separately on your own schedule and just call run_screens here).
  2. Runs both screens for today.
  3. Opens hypothetical positions for newly-flagged symbols (next day's
     open — logged as "pending", filled on the following day's run).
  4. Manages already-open paper positions: checks stop/target/trailing/time
     exit conditions against today's OHLCV and closes them if triggered.
  5. Appends all activity to the paper trading log CSV
     (config.paper_trade.log_file), matching the plan's schema:
     date_flagged, symbol, screen, entry_price, exit_price, exit_date,
     exit_reason, pnl_pct, notes.

This script is idempotent per calendar day — running it twice on the same
day without new data won't duplicate entries, but it does NOT re-fetch data
itself; pair it with a scheduled ingest_ohlcv run, or pass --ingest to do
both in one shot.

Usage:
    python -m paper_trade.daily_run
    python -m paper_trade.daily_run --ingest
"""
from __future__ import annotations

import argparse
import logging
from datetime import date

import pandas as pd

from config import CONFIG
from data_pipeline.storage import get_data_provider
from signals.run_screens import run_daily_screens

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LOG_COLUMNS = [
    "date_flagged", "symbol", "screen", "status", "entry_price", "entry_date",
    "exit_price", "exit_date", "exit_reason", "pnl_pct", "notes",
]


def _load_log() -> pd.DataFrame:
    path = CONFIG.paper_trade.log_file
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(path, parse_dates=["date_flagged", "entry_date", "exit_date"])


def _save_log(df: pd.DataFrame) -> None:
    path = CONFIG.paper_trade.log_file
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _open_new_positions(log_df: pd.DataFrame, shortlist: pd.DataFrame) -> pd.DataFrame:
    if shortlist.empty:
        return log_df

    already_logged = set(zip(log_df["symbol"], log_df["screen"])) if not log_df.empty else set()
    new_rows = []
    for _, row in shortlist.iterrows():
        key = (row["symbol"], row["screen_name"])
        if key in already_logged:
            continue
        new_rows.append({
            "date_flagged": row["date"],
            "symbol": row["symbol"],
            "screen": row["screen_name"],
            "status": "pending",  # filled on next day's open
            "entry_price": row["entry_price"],  # signal-day close, informational only
            "entry_date": pd.NaT,
            "exit_price": None,
            "exit_date": pd.NaT,
            "exit_reason": None,
            "pnl_pct": None,
            "notes": f"stop_loss={row['stop_loss']}; target_or_trailing={row['target_or_trailing_flag']}",
        })

    if not new_rows:
        return log_df

    logger.info("Opening %d new paper positions", len(new_rows))
    return pd.concat([log_df, pd.DataFrame(new_rows)], ignore_index=True)


def _manage_open_positions(log_df: pd.DataFrame, run_date: pd.Timestamp) -> pd.DataFrame:
    """Fill pending entries at today's open (if today > date_flagged), and
    check open positions for exit conditions using today's OHLCV."""
    if log_df.empty:
        return log_df

    provider = get_data_provider()

    for idx, position in log_df.iterrows():
        symbol = position["symbol"]
        df = provider.get_ohlcv(symbol, start=position["date_flagged"], end=run_date)
        if df.empty:
            continue
        df = df.sort_values("date").reset_index(drop=True)

        if position["status"] == "pending":
            fill_candidates = df[df["date"] > position["date_flagged"]]
            if fill_candidates.empty:
                continue
            fill_row = fill_candidates.iloc[0]
            log_df.at[idx, "status"] = "open"
            log_df.at[idx, "entry_date"] = fill_row["date"]
            log_df.at[idx, "entry_price"] = float(fill_row["open"])
            logger.info("Filled %s at %.2f on %s", symbol, fill_row["open"], fill_row["date"].date())
            continue

        if position["status"] != "open":
            continue

        notes = str(position["notes"])
        stop_loss = _parse_note_value(notes, "stop_loss")
        is_trailing = "target_or_trailing=trailing_stop" in notes

        since_entry = df[df["date"] > position["entry_date"]]
        if since_entry.empty:
            continue

        days_held = len(since_entry)
        entry_price = position["entry_price"]
        trailing_stop = stop_loss

        exit_triggered = False
        for _, day in since_entry.iterrows():
            if is_trailing and stop_loss is not None:
                initial_risk_pct = (entry_price - stop_loss) / entry_price
                candidate_stop = day["close"] * (1 - initial_risk_pct)
                trailing_stop = max(trailing_stop, candidate_stop)

            if stop_loss is not None and day["low"] <= trailing_stop:
                log_df.at[idx, "status"] = "closed"
                log_df.at[idx, "exit_date"] = day["date"]
                log_df.at[idx, "exit_price"] = trailing_stop
                log_df.at[idx, "exit_reason"] = "trailing" if is_trailing else "stop"
                exit_triggered = True
                break

        if not exit_triggered and days_held >= CONFIG.metrics.holding_period_days:
            last_day = since_entry.iloc[-1]
            log_df.at[idx, "status"] = "closed"
            log_df.at[idx, "exit_date"] = last_day["date"]
            log_df.at[idx, "exit_price"] = float(last_day["close"])
            log_df.at[idx, "exit_reason"] = "time"
            exit_triggered = True

        if exit_triggered:
            exit_price = log_df.at[idx, "exit_price"]
            log_df.at[idx, "pnl_pct"] = (exit_price - entry_price) / entry_price * 100
            logger.info(
                "Closed %s: entry=%.2f exit=%.2f pnl=%.2f%% reason=%s",
                symbol, entry_price, exit_price, log_df.at[idx, "pnl_pct"],
                log_df.at[idx, "exit_reason"],
            )

    return log_df


def _parse_note_value(notes: str, key: str) -> float | None:
    for part in notes.split(";"):
        part = part.strip()
        if part.startswith(f"{key}="):
            value = part.split("=", 1)[1]
            try:
                return float(value)
            except ValueError:
                return None
    return None


def run_paper_trading_day(run_date: pd.Timestamp | None = None) -> pd.DataFrame:
    run_date = run_date or pd.Timestamp(date.today())

    shortlist = run_daily_screens(run_date)

    log_df = _load_log()
    log_df = _open_new_positions(log_df, shortlist)
    log_df = _manage_open_positions(log_df, run_date)

    _save_log(log_df)
    logger.info("Paper trading log updated: %s", CONFIG.paper_trade.log_file)
    return log_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one day of paper trading")
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD, defaults to today")
    parser.add_argument(
        "--ingest", action="store_true",
        help="Also run OHLCV ingestion before screening (requires Dhan credentials)",
    )
    args = parser.parse_args()

    if args.ingest:
        from data_pipeline.ingest_ohlcv import main as ingest_main
        ingest_main()

    run_date = pd.to_datetime(args.date) if args.date else None
    run_paper_trading_day(run_date)


if __name__ == "__main__":
    main()
