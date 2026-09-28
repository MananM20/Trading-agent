"""
Seed the local NSE holiday calendar from the bundled CSV
(data_pipeline/universe/nse_holidays_seed.csv).

This seed only covers 2024-2026 (compiled from published NSE holiday
calendars as of Sep 2026). Re-run this after appending future years to the
seed file, or replace it with a call to whatever source you use to keep the
calendar current (NSE's own circulars are the authoritative source).

Usage:
    python -m data_pipeline.ingest_holidays
"""
from __future__ import annotations

import logging

import pandas as pd

from data_pipeline.storage import get_data_provider
from config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED_FILE = CONFIG.universe.universe_file.parent / "nse_holidays_seed.csv"


def main() -> None:
    if not SEED_FILE.exists():
        raise FileNotFoundError(f"Holiday seed file not found: {SEED_FILE}")

    df = pd.read_csv(SEED_FILE, parse_dates=["date"])
    holidays = df["date"].dt.date.tolist()

    provider = get_data_provider()
    provider.save_holiday_calendar(holidays)

    logger.info(
        "Saved %d NSE holidays (%s to %s) to %s",
        len(holidays),
        min(holidays),
        max(holidays),
        CONFIG.storage.holiday_calendar_file,
    )


if __name__ == "__main__":
    main()
