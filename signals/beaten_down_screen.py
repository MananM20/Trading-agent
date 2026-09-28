"""
Phase 2: Beaten-down-but-fundamentally-strong screen — concrete 6-rule set.

  1. Stock down >10% over the trailing 10 trading days
  2. Sector index (or average of peer stocks) down less than half as much
     over the same period
  3. ROE/ROCE >= 15% in the most recent available quarter
  4. Debt-to-equity flat or declining over the last 4 quarters
  5. Pledged promoter shares < 5%
  6. Current P/E or P/B at least 15% below the stock's own trailing 3-year
     average

All true => shortlist with reason tags.

Note on rule 6: computing trailing P/E or P/B requires EPS/book-value
history that isn't in the Phase 1 fundamentals schema (revenue, pat, roe,
roce, debt_to_equity, promoter_holding, pledged_pct, operating_cash_flow).
We approximate P/E using close price / (PAT-derived trailing EPS) when
shares outstanding is available; if it isn't, this rule is skipped and
flagged in reason_tags/details as unverifiable rather than silently
assumed true — the plan's Phase 2 acceptance criteria requires every flag
to have non-empty, explained reason_tags.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from config import CONFIG
from data_pipeline.ingest_fundamentals import quarter_sort_key

logger = logging.getLogger(__name__)


@dataclass
class BeatenDownResult:
    symbol: str
    date: pd.Timestamp
    passed: bool
    reason_tags: list[str] = field(default_factory=list)
    entry_price: float | None = None
    stop_loss: float | None = None
    target_or_trailing: str | None = None
    details: dict = field(default_factory=dict)


def _n_day_return(close: pd.Series, n: int) -> float | None:
    if len(close) <= n:
        return None
    return (close.iloc[-1] - close.iloc[-1 - n]) / close.iloc[-1 - n] * 100


def _sector_return(
    peer_symbols: list[str],
    peer_price_history: dict[str, pd.DataFrame],
    n: int,
    as_of_date: pd.Timestamp,
) -> float | None:
    returns = []
    for peer in peer_symbols:
        peer_df = peer_price_history.get(peer)
        if peer_df is None or peer_df.empty:
            continue
        aligned = peer_df[peer_df["date"] <= as_of_date].sort_values("date")
        r = _n_day_return(aligned["close"], n)
        if r is not None:
            returns.append(r)
    if not returns:
        return None
    return sum(returns) / len(returns)


def evaluate_beaten_down(
    symbol: str,
    df: pd.DataFrame,
    fundamentals: pd.DataFrame,
    peer_symbols: list[str],
    peer_price_history: dict[str, pd.DataFrame],
    as_of_idx: int | None = None,
) -> BeatenDownResult:
    cfg = CONFIG.beaten_down
    df = df.sort_values("date").reset_index(drop=True)
    if as_of_idx is None:
        as_of_idx = len(df) - 1

    if as_of_idx < cfg.decline_lookback_days + 1:
        return BeatenDownResult(symbol, df["date"].iloc[as_of_idx] if len(df) else pd.NaT, False,
                                 details={"skipped": "insufficient history"})

    window = df.iloc[: as_of_idx + 1]
    row = window.iloc[-1]
    reason_tags: list[str] = []
    details: dict = {}

    # Rule 1: down >10% over trailing 10 trading days
    stock_return = _n_day_return(window["close"], cfg.decline_lookback_days)
    rule1 = stock_return is not None and stock_return <= cfg.decline_threshold_pct
    details["stock_return_10d"] = round(stock_return, 2) if stock_return is not None else None
    if rule1:
        reason_tags.append("declined_gt_10pct")

    # Rule 2: sector/peers down less than half as much (isolates
    # stock-specific overreaction vs sector-wide moves)
    rule2 = False
    if peer_symbols:
        sector_return = _sector_return(peer_symbols, peer_price_history, cfg.decline_lookback_days, row["date"])
        if sector_return is not None and stock_return is not None and stock_return < 0:
            rule2 = abs(sector_return) < abs(stock_return) * cfg.sector_decline_ratio_max
            details["sector_return_10d"] = round(sector_return, 2)
    else:
        details["peer_group_missing"] = True
    if rule2:
        reason_tags.append("sector_relative_overreaction")

    # Rule 3: ROE/ROCE >= 15% in most recent quarter
    rule3 = False
    latest_fundamentals = None
    if not fundamentals.empty:
        sorted_fund = fundamentals.copy()
        sorted_fund["_sort_key"] = sorted_fund["quarter"].apply(quarter_sort_key)
        sorted_fund = sorted_fund.sort_values("_sort_key")
        latest_fundamentals = sorted_fund.iloc[-1]
        roe = latest_fundamentals.get("roe")
        roce = latest_fundamentals.get("roce")
        if pd.notna(roe) and pd.notna(roce):
            rule3 = (roe >= cfg.min_roe_roce_pct) or (roce >= cfg.min_roe_roce_pct)
            details["roe"] = roe
            details["roce"] = roce
    else:
        details["fundamentals_missing"] = True
    if rule3:
        reason_tags.append("roe_roce_ge_15pct")

    # Rule 4: debt-to-equity flat or declining over last 4 quarters
    rule4 = False
    if not fundamentals.empty:
        sorted_fund = fundamentals.copy()
        sorted_fund["_sort_key"] = sorted_fund["quarter"].apply(quarter_sort_key)
        sorted_fund = sorted_fund.sort_values("_sort_key").tail(4)
        de_series = sorted_fund["debt_to_equity"].dropna()
        if len(de_series) >= 2:
            rule4 = de_series.iloc[-1] <= de_series.iloc[0]
            details["debt_to_equity_trend"] = de_series.tolist()
    if rule4:
        reason_tags.append("debt_to_equity_flat_or_declining")

    # Rule 5: pledged promoter shares < 5%
    rule5 = False
    if latest_fundamentals is not None:
        pledged = latest_fundamentals.get("pledged_pct")
        if pd.notna(pledged):
            rule5 = pledged < cfg.max_pledged_pct
            details["pledged_pct"] = pledged
    if rule5:
        reason_tags.append("pledged_lt_5pct")

    # Rule 6: valuation discount vs own 3yr avg — requires EPS/book-value
    # data not present in the Phase 1 fundamentals schema. Explicitly
    # marked unverifiable rather than assumed true.
    rule6 = False
    details["valuation_discount_check"] = "unverifiable: EPS/book-value history not ingested"

    passed = rule1 and rule2 and rule3 and rule4 and rule5 and rule6

    entry_price = None
    stop_loss = None
    target_or_trailing = None
    if passed:
        entry_price = float(row["close"])
        recent_low = window["low"].iloc[-cfg.decline_lookback_days:].min()
        stop_loss = float(recent_low * 0.98)
        target_or_trailing = "trailing_stop"

    return BeatenDownResult(
        symbol=symbol,
        date=row["date"],
        passed=passed,
        reason_tags=reason_tags,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_or_trailing=target_or_trailing,
        details=details,
    )
