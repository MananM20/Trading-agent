"""
Phase 2: Breakout screen — concrete 5-rule set from the plan.

  1. Price closes above the most recent confirmed swing high
  2. Volume on breakout day >= 2x the 20-day average volume
  3. Delivery % on breakout day is in the top quartile of that stock's own
     trailing 60-day delivery % distribution
  4. RSI(14) is between 55-70 (momentum present but not already extreme)
  5. Stock's 5-day return minus Nifty 50's 5-day return > 0 (relative
     strength positive)

All 5 true => shortlist with reason tags.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from config import CONFIG
from signals.swing_detector import detect_fractal_swings, most_recent_confirmed_swing_high

logger = logging.getLogger(__name__)

try:
    import pandas_ta as ta
except ImportError:  # surfaced clearly when actually needed
    ta = None


@dataclass
class BreakoutResult:
    symbol: str
    date: pd.Timestamp
    passed: bool
    reason_tags: list[str] = field(default_factory=list)
    entry_price: float | None = None
    stop_loss: float | None = None
    target_or_trailing: str | None = None
    details: dict = field(default_factory=dict)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    if ta is not None:
        result = ta.rsi(close, length=period)
        return result if result is not None else pd.Series(index=close.index, dtype=float)

    # Fallback Wilder's RSI if pandas_ta isn't installed.
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def evaluate_breakout(
    symbol: str,
    df: pd.DataFrame,
    nifty_df: pd.DataFrame,
    as_of_idx: int | None = None,
) -> BreakoutResult:
    """Evaluate the breakout screen for one symbol as of a given row index
    (defaults to the last row = most recent trading day)."""
    cfg = CONFIG.breakout
    df = df.sort_values("date").reset_index(drop=True)
    if as_of_idx is None:
        as_of_idx = len(df) - 1

    min_rows_needed = max(cfg.delivery_lookback_days, cfg.volume_avg_window, cfg.rsi_period) + 5
    if as_of_idx < min_rows_needed:
        return BreakoutResult(symbol, df["date"].iloc[as_of_idx] if len(df) else pd.NaT, False,
                               details={"skipped": "insufficient history"})

    window = df.iloc[: as_of_idx + 1]
    row = window.iloc[-1]
    reason_tags: list[str] = []
    details: dict = {}

    # Rule 1: swing break
    swings = detect_fractal_swings(window.iloc[:-1])  # swings confirmed before today
    last_swing_high = most_recent_confirmed_swing_high(swings)
    rule1 = last_swing_high is not None and row["close"] > last_swing_high.price
    details["last_swing_high"] = last_swing_high.price if last_swing_high else None
    if rule1:
        reason_tags.append("swing_break")

    # Rule 2: volume >= 2x 20-day average (average excludes today to avoid
    # today's own spike inflating its own baseline)
    avg_volume = window["volume"].iloc[-(cfg.volume_avg_window + 1):-1].mean()
    volume_ratio = row["volume"] / avg_volume if avg_volume else 0
    rule2 = volume_ratio >= cfg.volume_multiple
    details["volume_ratio"] = round(float(volume_ratio), 2)
    if rule2:
        reason_tags.append("volume_2x")

    # Rule 3: delivery % in top quartile of trailing 60-day own distribution
    rule3 = False
    if "delivery_pct" in window.columns and pd.notna(row.get("delivery_pct")):
        trailing = window["delivery_pct"].iloc[-(cfg.delivery_lookback_days + 1):-1].dropna()
        if len(trailing) >= 10:
            threshold = trailing.quantile(cfg.delivery_top_quartile)
            rule3 = row["delivery_pct"] >= threshold
            details["delivery_pct_threshold"] = round(float(threshold), 2)
    if rule3:
        reason_tags.append("delivery_top_quartile")

    # Rule 4: RSI(14) between 55-70
    rsi_series = _rsi(window["close"], cfg.rsi_period)
    rsi_value = rsi_series.iloc[-1] if len(rsi_series) else None
    rule4 = rsi_value is not None and pd.notna(rsi_value) and cfg.rsi_low <= rsi_value <= cfg.rsi_high
    details["rsi"] = round(float(rsi_value), 2) if pd.notna(rsi_value) else None
    if rule4:
        reason_tags.append(f"rsi_{cfg.rsi_low}_{cfg.rsi_high}")

    # Rule 5: 5-day relative strength vs Nifty 50 > 0
    rule5 = False
    stock_5d_return = _n_day_return(window["close"], cfg.relative_strength_window)
    nifty_aligned = nifty_df[nifty_df["date"] <= row["date"]].sort_values("date")
    if stock_5d_return is not None and len(nifty_aligned) > cfg.relative_strength_window:
        nifty_5d_return = _n_day_return(nifty_aligned["close"], cfg.relative_strength_window)
        if nifty_5d_return is not None:
            rule5 = (stock_5d_return - nifty_5d_return) > 0
            details["relative_strength"] = round(float(stock_5d_return - nifty_5d_return), 3)
    if rule5:
        reason_tags.append("relative_strength_positive")

    passed = rule1 and rule2 and rule3 and rule4 and rule5

    entry_price = None
    stop_loss = None
    target_or_trailing = None
    if passed:
        entry_price = float(row["close"])
        if last_swing_high is not None:
            # Stop below the swing that was broken (structure-based stop).
            recent_low = window["low"].iloc[-10:].min()
            stop_loss = float(min(recent_low, entry_price * 0.95))
        from signals.swing_detector import compute_ath_regime
        ath_regime = compute_ath_regime(window)
        target_or_trailing = "trailing_stop" if ath_regime.near_ath else "next_swing_high"

    return BreakoutResult(
        symbol=symbol,
        date=row["date"],
        passed=passed,
        reason_tags=reason_tags,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_or_trailing=target_or_trailing,
        details=details,
    )


def _n_day_return(close: pd.Series, n: int) -> float | None:
    if len(close) <= n:
        return None
    return (close.iloc[-1] - close.iloc[-1 - n]) / close.iloc[-1 - n] * 100
