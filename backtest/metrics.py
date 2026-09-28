"""
Phase 3: Backtest metrics.

  - Win rate
  - Average win % vs average loss %
  - % of flagged stocks hitting +20% within 7 trading days (primary metric)
  - Max drawdown of a simulated equity curve
  - Sharpe ratio (annualized) and Sortino ratio
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import CONFIG


@dataclass
class BacktestMetrics:
    n_trades: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    pct_hit_target_within_horizon: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float

    def as_dict(self) -> dict:
        return {
            "n_trades": self.n_trades,
            "win_rate_pct": round(self.win_rate_pct, 2),
            "avg_win_pct": round(self.avg_win_pct, 2),
            "avg_loss_pct": round(self.avg_loss_pct, 2),
            "pct_hit_target_within_horizon": round(self.pct_hit_target_within_horizon, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
        }


def _equity_curve(pnl_pct_series: pd.Series, starting_capital: float = 1.0) -> pd.Series:
    """Equity curve assuming each trade risks a fixed fraction (here we
    just compound the pct returns sequentially — a simplification that
    assumes one position at a time / fixed capital allocation)."""
    growth = (1 + pnl_pct_series / 100)
    return starting_capital * growth.cumprod()


def _max_drawdown_pct(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max * 100
    return float(drawdown.min())


def _sharpe(pnl_pct_series: pd.Series, periods_per_year: int = 52) -> float:
    """Approximate annualized Sharpe using per-trade returns treated as a
    return series sampled roughly weekly (holding_period_days ~ 7)."""
    if len(pnl_pct_series) < 2 or pnl_pct_series.std() == 0:
        return 0.0
    mean_r = pnl_pct_series.mean() / 100
    std_r = pnl_pct_series.std() / 100
    return float((mean_r / std_r) * np.sqrt(periods_per_year))


def _sortino(pnl_pct_series: pd.Series, periods_per_year: int = 52) -> float:
    if len(pnl_pct_series) < 2:
        return 0.0
    mean_r = pnl_pct_series.mean() / 100
    downside = pnl_pct_series[pnl_pct_series < 0] / 100
    downside_std = downside.std() if len(downside) > 1 else 0.0
    if not downside_std:
        return 0.0
    return float((mean_r / downside_std) * np.sqrt(periods_per_year))


def compute_metrics(trades_df: pd.DataFrame) -> BacktestMetrics:
    if trades_df.empty:
        return BacktestMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    pnl = trades_df["pnl_pct"].dropna()
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    win_rate = len(wins) / len(pnl) * 100 if len(pnl) else 0.0
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0

    pct_hit_target = (
        trades_df["hit_target_within_horizon"].mean() * 100
        if "hit_target_within_horizon" in trades_df.columns and len(trades_df)
        else 0.0
    )

    equity = _equity_curve(pnl)
    max_dd = _max_drawdown_pct(equity)
    sharpe = _sharpe(pnl)
    sortino = _sortino(pnl)

    return BacktestMetrics(
        n_trades=len(trades_df),
        win_rate_pct=win_rate,
        avg_win_pct=float(avg_win),
        avg_loss_pct=float(avg_loss),
        pct_hit_target_within_horizon=float(pct_hit_target),
        max_drawdown_pct=max_dd,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
    )
