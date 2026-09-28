"""
Phase 2: Swing structure engine (rule-based, no ML).

Implements:
  - Fractal swing high/low detection (N bars either side)
  - Trend structure classification (Uptrend / Downtrend / Ranging) from the
    sequence of confirmed swings
  - ATH regime flag (near all-time-high vs not)

All operate on a single symbol's OHLCV dataframe, sorted by date ascending,
with at least columns: date, high, low, close.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from config import CONFIG


class SwingType(str, Enum):
    HIGH = "high"
    LOW = "low"


class TrendStructure(str, Enum):
    UPTREND = "uptrend"       # higher highs + higher lows
    DOWNTREND = "downtrend"   # lower highs + lower lows
    RANGING = "ranging"       # no clear sequential pattern


@dataclass
class Swing:
    date: pd.Timestamp
    price: float
    swing_type: SwingType
    confirmed: bool  # False for the most recent, still-forming swing


def detect_fractal_swings(df: pd.DataFrame, n: int | None = None) -> list[Swing]:
    """Fractal swing detection: a swing high is a bar whose high is greater
    than the N bars immediately before AND after it; swing low is the
    mirror on lows. The last N bars can't be confirmed yet (no bars after
    them), so they're returned with confirmed=False.

    df must be sorted by date ascending and indexed 0..len-1.
    """
    n = n or CONFIG.swing.fractal_n
    if len(df) < (2 * n + 1):
        return []

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    dates = pd.to_datetime(df["date"]).to_numpy()

    swings: list[Swing] = []
    last_confirmable_idx = len(df) - 1 - n

    for i in range(n, len(df) - n):
        window_highs = highs[i - n : i + n + 1]
        window_lows = lows[i - n : i + n + 1]

        is_swing_high = highs[i] == window_highs.max() and np.argmax(window_highs) == n
        is_swing_low = lows[i] == window_lows.min() and np.argmin(window_lows) == n

        confirmed = i <= last_confirmable_idx

        if is_swing_high:
            swings.append(Swing(pd.Timestamp(dates[i]), float(highs[i]), SwingType.HIGH, confirmed))
        if is_swing_low:
            swings.append(Swing(pd.Timestamp(dates[i]), float(lows[i]), SwingType.LOW, confirmed))

    swings.sort(key=lambda s: s.date)
    return swings


def detect_zigzag_swings(df: pd.DataFrame, pct_threshold: float | None = None) -> list[Swing]:
    """ZigZag alternative: register a new swing only once price has moved
    at least pct_threshold % from the last confirmed swing extreme. More
    robust to noise than the fractal method, but the most recent swing can
    repaint (change) as new bars arrive — the last swing returned is
    provisional (confirmed=False), all others are confirmed.
    """
    pct_threshold = pct_threshold or CONFIG.swing.zigzag_pct_threshold
    if df.empty:
        return []

    dates = pd.to_datetime(df["date"]).tolist()
    highs = df["high"].tolist()
    lows = df["low"].tolist()

    swings: list[Swing] = []
    # Seed: assume the first bar's high is a provisional swing high, we'll
    # correct direction as soon as a qualifying move happens.
    last_extreme_price = highs[0]
    last_extreme_date = dates[0]
    last_extreme_type = SwingType.HIGH
    direction: int | None = None  # +1 looking for next high, -1 looking for next low

    for i in range(1, len(df)):
        if direction is None:
            # Determine initial direction from first threshold-qualifying move.
            move_up = (highs[i] - last_extreme_price) / last_extreme_price * 100
            move_down = (last_extreme_price - lows[i]) / last_extreme_price * 100
            if move_down >= pct_threshold:
                swings.append(Swing(last_extreme_date, last_extreme_price, SwingType.HIGH, True))
                last_extreme_price, last_extreme_date, last_extreme_type = lows[i], dates[i], SwingType.LOW
                direction = 1
            elif move_up >= pct_threshold:
                direction = -1  # looking for a low to complete an up-then-down zigzag start
            continue

        if direction == 1:
            # Currently tracking down move from a high; look for it to
            # extend, or reverse by pct_threshold to confirm a low.
            if lows[i] < last_extreme_price:
                last_extreme_price, last_extreme_date = lows[i], dates[i]
            move_up = (highs[i] - last_extreme_price) / last_extreme_price * 100
            if move_up >= pct_threshold:
                swings.append(Swing(last_extreme_date, last_extreme_price, SwingType.LOW, True))
                last_extreme_price, last_extreme_date, last_extreme_type = highs[i], dates[i], SwingType.HIGH
                direction = -1
        else:
            if highs[i] > last_extreme_price:
                last_extreme_price, last_extreme_date = highs[i], dates[i]
            move_down = (last_extreme_price - lows[i]) / last_extreme_price * 100
            if move_down >= pct_threshold:
                swings.append(Swing(last_extreme_date, last_extreme_price, SwingType.HIGH, True))
                last_extreme_price, last_extreme_date, last_extreme_type = lows[i], dates[i], SwingType.LOW
                direction = 1

    # The current in-progress extreme (not yet confirmed by a reversal) is provisional.
    swings.append(Swing(last_extreme_date, last_extreme_price, last_extreme_type, False))
    return swings


def classify_trend_structure(
    swings: list[Swing], lookback: int | None = None
) -> TrendStructure:
    """Classify trend from the sequence of confirmed swing highs/lows.

    Uptrend: each confirmed swing high is higher than the previous swing
    high AND each confirmed swing low is higher than the previous swing low
    (HH + HL). Downtrend is the mirror (LH + LL). Anything else -> Ranging.
    """
    lookback = lookback or CONFIG.swing.trend_lookback_swings
    confirmed = [s for s in swings if s.confirmed]
    recent = confirmed[-lookback:] if len(confirmed) > lookback else confirmed

    swing_highs = [s.price for s in recent if s.swing_type == SwingType.HIGH]
    swing_lows = [s.price for s in recent if s.swing_type == SwingType.LOW]

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return TrendStructure.RANGING

    highs_rising = all(b > a for a, b in zip(swing_highs, swing_highs[1:]))
    highs_falling = all(b < a for a, b in zip(swing_highs, swing_highs[1:]))
    lows_rising = all(b > a for a, b in zip(swing_lows, swing_lows[1:]))
    lows_falling = all(b < a for a, b in zip(swing_lows, swing_lows[1:]))

    if highs_rising and lows_rising:
        return TrendStructure.UPTREND
    if highs_falling and lows_falling:
        return TrendStructure.DOWNTREND
    return TrendStructure.RANGING


def most_recent_confirmed_swing_high(swings: list[Swing]) -> Swing | None:
    highs = [s for s in swings if s.swing_type == SwingType.HIGH and s.confirmed]
    return highs[-1] if highs else None


def most_recent_confirmed_swing_low(swings: list[Swing]) -> Swing | None:
    lows = [s for s in swings if s.swing_type == SwingType.LOW and s.confirmed]
    return lows[-1] if lows else None


@dataclass
class AthRegime:
    all_time_high: float
    current_close: float
    distance_from_ath_pct: float
    near_ath: bool


def compute_ath_regime(df: pd.DataFrame, threshold_pct: float | None = None) -> AthRegime:
    """distance_from_ath_pct = (ath - close) / ath * 100.
    near_ath=True switches exit logic to trailing-stop (no fixed target)
    per the plan; otherwise use next-swing-high as target.
    """
    threshold_pct = threshold_pct or CONFIG.swing.near_ath_pct_threshold
    ath = float(df["high"].max())
    current_close = float(df["close"].iloc[-1])
    distance_pct = (ath - current_close) / ath * 100 if ath > 0 else 0.0
    return AthRegime(
        all_time_high=ath,
        current_close=current_close,
        distance_from_ath_pct=distance_pct,
        near_ath=distance_pct < threshold_pct,
    )


def analyze_symbol(df: pd.DataFrame) -> dict:
    """Convenience wrapper: run fractal swings + trend classification + ATH
    regime for one symbol's OHLCV dataframe. Returns a dict summary."""
    df = df.sort_values("date").reset_index(drop=True)
    swings = detect_fractal_swings(df)
    trend = classify_trend_structure(swings)
    ath_regime = compute_ath_regime(df)
    last_swing_high = most_recent_confirmed_swing_high(swings)
    last_swing_low = most_recent_confirmed_swing_low(swings)

    return {
        "trend_structure": trend,
        "swings": swings,
        "last_confirmed_swing_high": last_swing_high,
        "last_confirmed_swing_low": last_swing_low,
        "ath_regime": ath_regime,
    }
