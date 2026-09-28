"""
Phase 3: Sanity baseline — random stock selection on the same days, same
universe, same holding period/target, to check whether a screen actually
beats picking stocks at random.
"""
from __future__ import annotations

import random

import pandas as pd

from config import CONFIG
from backtest.engine import Trade, simulate_trade
from backtest.metrics import BacktestMetrics, compute_metrics


def run_random_baseline(
    signal_dates: list[pd.Timestamp],
    n_picks_per_date: int,
    universe_symbols: list[str],
    price_history: dict[str, pd.DataFrame],
    large_cap_symbols: set[str],
    n_trials: int | None = None,
    seed: int = 42,
) -> tuple[BacktestMetrics, list[BacktestMetrics]]:
    """Run n_trials random-selection simulations and return the averaged
    metrics plus the per-trial list (so a distribution/spread can be shown,
    not just a point estimate)."""
    cfg = CONFIG.backtest
    n_trials = n_trials or cfg.random_baseline_trials
    rng = random.Random(seed)

    trial_metrics: list[BacktestMetrics] = []

    for trial in range(n_trials):
        trades: list[Trade] = []
        for signal_date in signal_dates:
            eligible = [s for s in universe_symbols if s in price_history and not price_history[s].empty]
            if not eligible:
                continue
            picks = rng.sample(eligible, min(n_picks_per_date, len(eligible)))
            for symbol in picks:
                df = price_history[symbol]
                past = df[df["date"] <= signal_date]
                if past.empty:
                    continue
                entry_price = float(past["close"].iloc[-1])
                stop_loss = entry_price * 0.95  # generic 5% stop for the baseline
                trade = simulate_trade(
                    symbol=symbol,
                    screen_name="random_baseline",
                    signal_date=signal_date,
                    signal_entry_price=entry_price,
                    stop_loss=stop_loss,
                    is_trailing=False,
                    target_price=entry_price * (1 + CONFIG.metrics.target_return_pct / 100),
                    price_df=df,
                    holding_period_days=CONFIG.metrics.holding_period_days,
                    target_return_pct=CONFIG.metrics.target_return_pct,
                    large_cap_symbols=large_cap_symbols,
                )
                trades.append(trade)

        trades_df = pd.DataFrame([t.__dict__ for t in trades])
        trial_metrics.append(compute_metrics(trades_df))

    avg_metrics = BacktestMetrics(
        n_trades=int(sum(m.n_trades for m in trial_metrics) / max(len(trial_metrics), 1)),
        win_rate_pct=sum(m.win_rate_pct for m in trial_metrics) / max(len(trial_metrics), 1),
        avg_win_pct=sum(m.avg_win_pct for m in trial_metrics) / max(len(trial_metrics), 1),
        avg_loss_pct=sum(m.avg_loss_pct for m in trial_metrics) / max(len(trial_metrics), 1),
        pct_hit_target_within_horizon=sum(m.pct_hit_target_within_horizon for m in trial_metrics) / max(len(trial_metrics), 1),
        max_drawdown_pct=sum(m.max_drawdown_pct for m in trial_metrics) / max(len(trial_metrics), 1),
        sharpe_ratio=sum(m.sharpe_ratio for m in trial_metrics) / max(len(trial_metrics), 1),
        sortino_ratio=sum(m.sortino_ratio for m in trial_metrics) / max(len(trial_metrics), 1),
    )

    return avg_metrics, trial_metrics
