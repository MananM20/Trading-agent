"""
Phase 3: Backtest runner — the make-or-break gate before Phase 4/5/6.

For each historical trading day in the lookback window, re-runs both
screens (using only data available up to and including that day — no
look-ahead), simulates each resulting signal as a trade entered at the
NEXT day's open, and aggregates metrics per-screen and combined, compared
against a random-selection baseline.

LIMITATIONS (read before trusting the report):
  1. Survivorship bias is NOT solved: the universe list is the CURRENT
     Nifty-500-style constituent list (data_pipeline/universe/nifty500.csv),
     not a historical point-in-time list. Stocks that were delisted, merged,
     or renamed during the backtest window are excluded, which optimistically
     biases every metric here versus a true historical run. Source a
     historical constituent list before trusting these numbers for capital
     allocation decisions.
  2. Costs are approximate: brokerage_pct in engine.py assumes a rough
     ~100-share trade size to convert Dhan's flat per-order brokerage into
     a percentage; for very different position sizes the pct drag will be
     off. Tune config.backtest before relying on absolute PnL figures.
  3. The beaten-down screen's rule 6 (valuation discount vs 3yr average)
     is structurally unverifiable with the current fundamentals schema
     (see signals/beaten_down_screen.py), so that screen will currently
     produce zero signals until the schema is extended with EPS/book-value
     history.

Usage:
    python -m backtest.run_backtest --years 2
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta

import pandas as pd

from config import CONFIG
from data_pipeline.storage import get_data_provider
from signals.beaten_down_screen import evaluate_beaten_down
from signals.breakout_screen import evaluate_breakout
from backtest.engine import Trade, simulate_trade
from backtest.metrics import compute_metrics
from backtest.random_baseline import run_random_baseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NIFTY50_SYMBOL = "NIFTY50"


def generate_signals(
    universe_symbols: list[str],
    price_history: dict[str, pd.DataFrame],
    nifty_df: pd.DataFrame,
    peer_groups: dict[str, str],
    fundamentals_cache: dict[str, pd.DataFrame],
    trading_days: list[pd.Timestamp],
) -> pd.DataFrame:
    """Walk forward day by day, evaluating both screens with only data up
    to and including that day, collecting every (symbol, date) that fires
    for the first time (Phase 0's "flagged once" rule)."""
    rows = []
    already_flagged: dict[str, set[str]] = {
        CONFIG.screens.breakout: set(),
        CONFIG.screens.beaten_down: set(),
    }

    for day in trading_days:
        for symbol in universe_symbols:
            df = price_history.get(symbol)
            if df is None or df.empty:
                continue
            window = df[df["date"] <= day]
            if window.empty or window["date"].iloc[-1] != day:
                continue  # no trading data for this symbol on this day

            if symbol not in already_flagged[CONFIG.screens.breakout]:
                result = evaluate_breakout(symbol, window, nifty_df)
                if result.passed:
                    already_flagged[CONFIG.screens.breakout].add(symbol)
                    rows.append({
                        "date": day, "symbol": symbol,
                        "screen_name": CONFIG.screens.breakout,
                        "entry_price": result.entry_price,
                        "stop_loss": result.stop_loss,
                        "is_trailing": result.target_or_trailing == "trailing_stop",
                        "target_price": (
                            None if result.target_or_trailing == "trailing_stop"
                            else result.details.get("last_swing_high")
                        ),
                    })

            if symbol not in already_flagged[CONFIG.screens.beaten_down]:
                peers_raw = peer_groups.get(symbol, "")
                peer_symbols = [p for p in str(peers_raw).split("|") if p] if pd.notna(peers_raw) else []
                fundamentals = fundamentals_cache.get(symbol, pd.DataFrame())
                result = evaluate_beaten_down(symbol, window, fundamentals, peer_symbols, price_history)
                if result.passed:
                    already_flagged[CONFIG.screens.beaten_down].add(symbol)
                    rows.append({
                        "date": day, "symbol": symbol,
                        "screen_name": CONFIG.screens.beaten_down,
                        "entry_price": result.entry_price,
                        "stop_loss": result.stop_loss,
                        "is_trailing": True,
                        "target_price": None,
                    })

    return pd.DataFrame(rows)


def run_backtest(years: int) -> dict:
    provider = get_data_provider()
    universe = provider.get_universe()
    universe_symbols = universe["symbol"].tolist()
    large_cap_symbols = set(universe.loc[universe["market_cap_category"] == "Large", "symbol"])
    peer_groups = provider.get_peer_groups().set_index("symbol")["peers"].to_dict()

    end = date.today()
    start = end - timedelta(days=365 * years)

    price_history = {s: provider.get_ohlcv(s, start, end) for s in universe_symbols}
    price_history = {s: df for s, df in price_history.items() if not df.empty}
    nifty_df = provider.get_ohlcv(NIFTY50_SYMBOL, start, end)
    fundamentals_cache = {s: provider.get_fundamentals(s) for s in universe_symbols}

    if not price_history:
        raise RuntimeError(
            "No OHLCV data found for any universe symbol. Run "
            "`python -m data_pipeline.ingest_ohlcv` first."
        )

    all_dates = sorted(set().union(*[set(df["date"]) for df in price_history.values()]))

    logger.info("Generating signals across %d trading days...", len(all_dates))
    signals_df = generate_signals(
        universe_symbols, price_history, nifty_df, peer_groups, fundamentals_cache, all_dates
    )
    logger.info("Generated %d signals total", len(signals_df))

    trades: list[Trade] = []
    for _, sig in signals_df.iterrows():
        df = price_history[sig["symbol"]]
        trade = simulate_trade(
            symbol=sig["symbol"],
            screen_name=sig["screen_name"],
            signal_date=sig["date"],
            signal_entry_price=sig["entry_price"],
            stop_loss=sig["stop_loss"],
            is_trailing=sig["is_trailing"],
            target_price=sig["target_price"],
            price_df=df,
            holding_period_days=CONFIG.metrics.holding_period_days,
            target_return_pct=CONFIG.metrics.target_return_pct,
            large_cap_symbols=large_cap_symbols,
        )
        trades.append(trade)

    trades_df = pd.DataFrame([t.__dict__ for t in trades])

    results = {"signals": signals_df, "trades": trades_df}

    for screen_name in (CONFIG.screens.breakout, CONFIG.screens.beaten_down):
        subset = trades_df[trades_df["screen_name"] == screen_name] if not trades_df.empty else trades_df
        results[f"metrics_{screen_name}"] = compute_metrics(subset)

    results["metrics_combined"] = compute_metrics(trades_df)

    if not signals_df.empty:
        signal_dates = signals_df["date"].unique().tolist()
        avg_picks_per_date = max(1, round(len(signals_df) / max(len(signal_dates), 1)))
        baseline_avg, baseline_trials = run_random_baseline(
            signal_dates=signal_dates,
            n_picks_per_date=avg_picks_per_date,
            universe_symbols=universe_symbols,
            price_history=price_history,
            large_cap_symbols=large_cap_symbols,
        )
        results["metrics_random_baseline"] = baseline_avg
    else:
        results["metrics_random_baseline"] = None

    return results


def render_report(results: dict, years: int) -> str:
    lines = ["# Backtest Report", ""]
    lines.append(f"Lookback window: {years} years. Generated by `backtest/run_backtest.py`.")
    lines.append("")
    lines.append(
        "**Limitations**: universe is the current constituent list (survivorship bias, "
        "not corrected), cost model is approximate, and the beaten-down screen's "
        "valuation-discount rule is currently unverifiable (missing EPS/book-value data) "
        "so it will not produce signals. See run_backtest.py module docstring for detail."
    )
    lines.append("")

    def metrics_table(name: str, metrics) -> list[str]:
        if metrics is None:
            return [f"## {name}", "", "No signals generated — nothing to report.", ""]
        d = metrics.as_dict()
        out = [f"## {name}", "", "| Metric | Value |", "|---|---|"]
        for k, v in d.items():
            out.append(f"| {k} | {v} |")
        out.append("")
        return out

    lines += metrics_table(f"Breakout screen ({CONFIG.screens.breakout})", results.get(f"metrics_{CONFIG.screens.breakout}"))
    lines += metrics_table(f"Beaten-down screen ({CONFIG.screens.beaten_down})", results.get(f"metrics_{CONFIG.screens.beaten_down}"))
    lines += metrics_table("Combined (both screens)", results.get("metrics_combined"))
    lines += metrics_table(f"Random baseline (avg of {CONFIG.backtest.random_baseline_trials} trials)", results.get("metrics_random_baseline"))

    combined = results.get("metrics_combined")
    baseline = results.get("metrics_random_baseline")
    if combined and baseline and baseline.n_trades:
        edge = combined.pct_hit_target_within_horizon - baseline.pct_hit_target_within_horizon
        lines.append("## Edge vs random baseline")
        lines.append("")
        lines.append(
            f"Combined screens hit the primary metric "
            f"({CONFIG.metrics.target_return_pct}% within {CONFIG.metrics.holding_period_days} days) "
            f"{combined.pct_hit_target_within_horizon:.2f}% of the time, vs "
            f"{baseline.pct_hit_target_within_horizon:.2f}% for random selection "
            f"({edge:+.2f} percentage points)."
        )
        lines.append("")
        if edge <= 0:
            lines.append(
                "**This screen does not beat random selection on the primary metric.** "
                "Per the plan's Phase 3 acceptance criteria, this means no statistical "
                "edge has been demonstrated — do not proceed to Phase 4 paper trading "
                "on this basis alone."
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 backtest")
    parser.add_argument("--years", type=int, default=CONFIG.backtest.lookback_years)
    args = parser.parse_args()

    results = run_backtest(args.years)
    report = render_report(results, args.years)

    report_path = CONFIG.storage.ohlcv_dir.parent / "backtest_report.md"
    report_path.write_text(report, encoding="utf-8")

    trades_path = CONFIG.storage.ohlcv_dir.parent / "backtest_trades.csv"
    results["trades"].to_csv(trades_path, index=False)

    logger.info("Backtest report written to %s", report_path)
    logger.info("Trade log written to %s", trades_path)
    print(report)


if __name__ == "__main__":
    main()
