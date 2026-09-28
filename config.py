"""
Central configuration for the swing-trading agent.

Every other phase reads from this file — nothing about universe, capital,
risk, holding period, or screen names should be hardcoded elsewhere.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"

DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Phase 0 decisions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UniverseConfig:
    # Source of the tradable universe. "nifty500" ships with a CSV under
    # data_pipeline/universe/nifty500.csv (user-maintained / refreshed
    # periodically from NSE). Swap to "all_nse" later without touching
    # ingestion code — just point this at a different list.
    source: str = "nifty500"
    universe_file: Path = PROJECT_ROOT / "data_pipeline" / "universe" / "nifty500.csv"
    # Optional peer-group mapping for the beaten-down screen (sector index
    # or 5 peer stocks per symbol). CSV columns: symbol, sector, peers (pipe
    # separated).
    peer_group_file: Path = PROJECT_ROOT / "data_pipeline" / "universe" / "peer_groups.csv"


@dataclass(frozen=True)
class CapitalConfig:
    capital_inr: float = float(os.getenv("TRADING_CAPITAL_INR", "1000000"))
    risk_pct_per_trade: float = float(os.getenv("RISK_PCT_PER_TRADE", "0.02"))  # 2%

    def position_size(self, entry_price: float, stop_loss_price: float) -> int:
        """Shares to buy given the fixed risk-per-trade rule.

        position_size = (risk_pct * capital) / (entry - stop_loss)
        Returns whole shares (floored), 0 if risk-per-share is non-positive.
        """
        risk_amount = self.capital_inr * self.risk_pct_per_trade
        risk_per_share = entry_price - stop_loss_price
        if risk_per_share <= 0:
            return 0
        return int(risk_amount // risk_per_share)


@dataclass(frozen=True)
class MetricsConfig:
    # Phase 0: pick ONE primary metric to optimize for.
    primary_metric: str = "pct_flagged_hitting_target_within_horizon"
    target_return_pct: float = 20.0
    holding_period_days: int = 7  # trading days
    # Secondary/guardrail metrics — reported but not optimized for.
    secondary_metrics: tuple[str, ...] = ("sharpe_ratio", "sortino_ratio", "max_drawdown")


@dataclass(frozen=True)
class ScreenNames:
    breakout: str = "breakout"
    beaten_down: str = "beaten_down_strong"


@dataclass(frozen=True)
class SwingConfig:
    # Fractal swing detection: bars on each side that must be lower/higher.
    fractal_n: int = 2
    # ZigZag alternative minimum % move to register a new swing.
    zigzag_pct_threshold: float = 4.0
    # Number of recent confirmed swings used to classify trend structure.
    trend_lookback_swings: int = 6
    # "Near all-time-high" regime threshold.
    near_ath_pct_threshold: float = 2.0


@dataclass(frozen=True)
class BreakoutScreenConfig:
    volume_multiple: float = 2.0
    volume_avg_window: int = 20
    delivery_lookback_days: int = 60
    delivery_top_quartile: float = 0.75  # percentile threshold
    rsi_period: int = 14
    rsi_low: float = 55.0
    rsi_high: float = 70.0
    relative_strength_window: int = 5  # trading days


@dataclass(frozen=True)
class BeatenDownScreenConfig:
    decline_lookback_days: int = 10
    decline_threshold_pct: float = -10.0
    sector_decline_ratio_max: float = 0.5  # stock decline vs sector decline
    min_roe_roce_pct: float = 15.0
    max_pledged_pct: float = 5.0
    valuation_discount_vs_3yr_avg_pct: float = 15.0


@dataclass(frozen=True)
class DataQualityConfig:
    max_single_day_move_pct: float = 15.0  # corporate-action-suspect threshold
    max_missing_day_pct: float = 1.0  # acceptance criterion for Phase 1
    delivery_pct_min: float = 0.0
    delivery_pct_max: float = 100.0


@dataclass(frozen=True)
class BacktestConfig:
    lookback_years: int = 2
    brokerage_flat_inr: float = 20.0  # per executed order leg, adjust to Dhan's plan
    stt_delivery_sell_pct: float = 0.1
    slippage_large_cap_pct: float = 0.1
    slippage_mid_small_cap_pct: float = 0.2
    random_baseline_trials: int = 200  # number of random-selection simulations


@dataclass(frozen=True)
class PaperTradeConfig:
    min_duration_weeks: int = 4
    max_duration_weeks: int = 8
    log_file: Path = LOG_DIR / "paper_trade_log.csv"


@dataclass(frozen=True)
class DhanConfig:
    client_id: str = os.getenv("DHAN_CLIENT_ID", "")
    access_token: str = os.getenv("DHAN_ACCESS_TOKEN", "")
    api_key: str = os.getenv("DHAN_API_KEY", "")
    # Conservative default; confirm against Dhan's current published limits
    # before increasing.
    requests_per_second: float = float(os.getenv("DHAN_REQUESTS_PER_SECOND", "5"))


@dataclass(frozen=True)
class StorageConfig:
    # "parquet" needs no external DB. Switch to "postgres" later if desired;
    # ingestion/signal code should only talk to the DataProvider interface.
    backend: str = os.getenv("STORAGE_BACKEND", "parquet")
    ohlcv_dir: Path = DATA_DIR / "eod"
    fundamentals_dir: Path = DATA_DIR / "fundamentals"
    shortlist_dir: Path = DATA_DIR / "shortlists"
    holiday_calendar_file: Path = DATA_DIR / "nse_holidays.csv"
    postgres_dsn: str = os.getenv("POSTGRES_DSN", "")


@dataclass(frozen=True)
class Config:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    capital: CapitalConfig = field(default_factory=CapitalConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    screens: ScreenNames = field(default_factory=ScreenNames)
    swing: SwingConfig = field(default_factory=SwingConfig)
    breakout: BreakoutScreenConfig = field(default_factory=BreakoutScreenConfig)
    beaten_down: BeatenDownScreenConfig = field(default_factory=BeatenDownScreenConfig)
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    paper_trade: PaperTradeConfig = field(default_factory=PaperTradeConfig)
    dhan: DhanConfig = field(default_factory=DhanConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)


CONFIG = Config()
