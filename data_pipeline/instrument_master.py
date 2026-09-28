"""
Resolve NSE trading symbols (e.g. "RELIANCE") to Dhan's `security_id`.

Dhan's historical data API requires a security_id, not the plain trading
symbol, so ingestion needs this mapping. Dhan publishes a CSV scrip master:
https://images.dhan.co/api-data/api-scrip-master-detailed.csv
(documented at https://dhanhq.co/docs/v2/instruments/)

The file is large (all exchanges/segments/derivatives) and changes rarely
outside of new listings/expiries, so we cache it locally and refresh on
demand rather than downloading on every run.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from config import CONFIG

logger = logging.getLogger(__name__)

SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"
CACHE_PATH: Path = CONFIG.storage.ohlcv_dir.parent / "scrip_master.csv"


def refresh_scrip_master(force: bool = False) -> Path:
    """Download and cache the Dhan scrip master CSV."""
    if CACHE_PATH.exists() and not force:
        return CACHE_PATH

    logger.info("Downloading Dhan scrip master from %s", SCRIP_MASTER_URL)
    resp = requests.get(SCRIP_MASTER_URL, timeout=60)
    resp.raise_for_status()

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_bytes(resp.content)
    logger.info("Cached scrip master to %s", CACHE_PATH)
    return CACHE_PATH


def load_scrip_master() -> pd.DataFrame:
    path = refresh_scrip_master()
    return pd.read_csv(path, low_memory=False)


def build_symbol_to_security_id_map(
    exchange: str = "NSE",
    segment: str = "E",
) -> dict[str, str]:
    """Map trading symbol -> security_id for NSE cash-equity instruments.

    segment "E" = Equity (per Dhan's SEGMENT column: C/D/E/M).
    """
    df = load_scrip_master()

    exch_col = "EXCH_ID" if "EXCH_ID" in df.columns else "SEM_EXM_EXCH_ID"
    seg_col = "SEGMENT" if "SEGMENT" in df.columns else "SEM_SEGMENT"
    symbol_col = "SEM_TRADING_SYMBOL" if "SEM_TRADING_SYMBOL" in df.columns else "SYMBOL_NAME"
    secid_col = "SEM_SMST_SECURITY_ID" if "SEM_SMST_SECURITY_ID" in df.columns else "SECURITY_ID"

    missing = [c for c in (exch_col, seg_col, symbol_col, secid_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Scrip master is missing expected columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    filtered = df[(df[exch_col] == exchange) & (df[seg_col] == segment)]
    mapping = dict(zip(filtered[symbol_col].astype(str), filtered[secid_col].astype(str)))
    return mapping


def get_security_id(symbol: str, mapping: dict[str, str] | None = None) -> str:
    mapping = mapping or build_symbol_to_security_id_map()
    if symbol not in mapping:
        raise KeyError(
            f"Symbol '{symbol}' not found in NSE equity scrip master. "
            f"Check spelling or run refresh_scrip_master(force=True)."
        )
    return mapping[symbol]
