"""
Phase 3: Event-driven backtest engine.

A custom lightweight backtester rather than vectorbt: the plan's exit logic
(target / stop / trailing / time-based) is inherently path-dependent per
trade, which doesn't map cleanly onto vectorbt's vectorized signal-array
model. This keeps the logic simple, auditable, and dependency-light.

Bias controls (mandatory per the plan):
  - No look-ahead: a signal fired on day T executes at day T+1's open.
  - Realistic costs: brokerage (flat per leg) + STT on delivery sell +
    slippage assumption, all from config.backtest.
  - Survivorship bias: NOT fully solved here. See LIMITATIONS.md /
    README — this backtester only has access to the current universe
    list (data_pipeline/universe/nifty500.csv), so delisted/renamed
    stocks during the backtest window are excluded. Results are
    optimistic versus reality to an unknown degree until a historical
    constituent list is sourced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from config import CONFIG


class ExitReason(str, Enum):
    TARGET = "target"
    STOP = "stop"
    TRAILING = "trailing"
    TIME = "time"
    NO_DATA = "no_data"


@dataclass
class Trade:
    symbol: str
    screen_name: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    stop_loss: float
    target_price: float | None
    is_trailing: bool
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: ExitReason | None = None
    pnl_pct: float | None = None
    hit_target_within_horizon: bool | None = None
    days_held: int | None = None


def _slippage_pct(symbol: str, large_cap_symbols: set[str]) -> float:
    cfg = CONFIG.backtest
    return cfg.slippage_large_cap_pct if symbol in large_cap_symbols else cfg.slippage_mid_small_cap_pct


def _apply_costs(entry_price: float, exit_price: float, symbol: str, large_cap_symbols: set[str]) -> tuple[float, float]:
    """Apply slippage to entry/exit fills and return adjusted prices.
    Brokerage/STT are applied separately as a pnl_pct deduction."""
    slip = _slippage_pct(symbol, large_cap_symbols) / 100
    adj_entry = entry_price * (1 + slip)   # buy fills slightly worse (higher)
    adj_exit = exit_price * (1 - slip)     # sell fills slightly worse (lower)
    return adj_entry, adj_exit


def simulate_trade(
    symbol: str,
    screen_name: str,
    signal_date: pd.Timestamp,
    signal_entry_price: float,
    stop_loss: float,
    is_trailing: bool,
    target_price: float | None,
    price_df: pd.DataFrame,
    holding_period_days: int,
    target_return_pct: float,
    large_cap_symbols: set[str],
) -> Trade:
    """Simulate one trade: enter at next day's open after the signal,
    then walk forward day by day applying stop/target/trailing/time exit
    rules, until `holding_period_days` trading days have elapsed."""
    cfg = CONFIG.backtest

    future = price_df[price_df["date"] > signal_date].sort_values("date").reset_index(drop=True)
    if future.empty:
        return Trade(symbol, screen_name, signal_date, signal_date, signal_entry_price,
                     stop_loss, target_price, is_trailing, exit_reason=ExitReason.NO_DATA)

    entry_row = future.iloc[0]
    entry_date = entry_row["date"]
    entry_price = float(entry_row["open"])  # next-day-open execution, no look-ahead

    trailing_stop = stop_loss
    exit_date = None
    exit_price = None
    exit_reason = None
    hit_target_within_horizon = False

    horizon = future.iloc[1 : holding_period_days + 1] if len(future) > 1 else future.iloc[0:0]

    for _, day in horizon.iterrows():
        if is_trailing:
            # Trail the stop up to close_price * (1 - initial_risk_pct), never down.
            initial_risk_pct = (entry_price - stop_loss) / entry_price
            candidate_stop = day["close"] * (1 - initial_risk_pct)
            trailing_stop = max(trailing_stop, candidate_stop)

        day_return_pct = (day["close"] - entry_price) / entry_price * 100
        if day_return_pct >= target_return_pct:
            hit_target_within_horizon = True

        if day["low"] <= trailing_stop:
            exit_date = day["date"]
            exit_price = trailing_stop
            exit_reason = ExitReason.TRAILING if is_trailing else ExitReason.STOP
            break

        if not is_trailing and target_price is not None and day["high"] >= target_price:
            exit_date = day["date"]
            exit_price = target_price
            exit_reason = ExitReason.TARGET
            break

    if exit_date is None:
        # Time exit: close out at the close of the last bar in the horizon.
        last_bar = horizon.iloc[-1] if len(horizon) else entry_row
        exit_date = last_bar["date"]
        exit_price = float(last_bar["close"])
        exit_reason = ExitReason.TIME

    adj_entry, adj_exit = _apply_costs(entry_price, float(exit_price), symbol, large_cap_symbols)

    gross_pnl_pct = (adj_exit - adj_entry) / adj_entry * 100
    # Cost drag: brokerage (2 legs, as pct of notional) + STT on sell leg.
    brokerage_pct = (cfg.brokerage_flat_inr * 2 / (adj_entry * 100)) * 100  # rough, assumes ~100 shares; see NOTE below
    stt_pct = cfg.stt_delivery_sell_pct
    net_pnl_pct = gross_pnl_pct - stt_pct - brokerage_pct

    days_held = (pd.to_datetime(exit_date) - pd.to_datetime(entry_date)).days

    return Trade(
        symbol=symbol,
        screen_name=screen_name,
        signal_date=signal_date,
        entry_date=entry_date,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_price=target_price,
        is_trailing=is_trailing,
        exit_date=exit_date,
        exit_price=float(exit_price),
        exit_reason=exit_reason,
        pnl_pct=net_pnl_pct,
        hit_target_within_horizon=hit_target_within_horizon,
        days_held=days_held,
    )
