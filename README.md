# Swing-Trading Agent (Indian Equities)

Implementation of `trading-agent-plan.md`, Phases 0-4: config, local data
pipeline (Dhan + NSE), rule-based swing/breakout/beaten-down screens,
event-driven backtester, and local paper trading. Options (Phase 5) and ML
(Phase 6) are not built — per the plan, they only make sense after Phase
3/4 demonstrate an edge.

## Before you rely on this for real capital

Read this section. Several gaps and deviations from the original plan are
built in on purpose, not bugs:

1. **The universe file is a placeholder.** `data_pipeline/universe/nifty500.csv`
   contains ~40 large-cap names, not the real Nifty 500 constituent list.
   Replace it with an actual current constituent export (same columns:
   `symbol,exchange,sector,market_cap_category`) before running anything
   for real. `data_pipeline/universe/peer_groups.csv` needs the same
   treatment for the beaten-down screen's sector-relative rule.

2. **The beaten-down screen currently cannot pass.** Its rule 6 ("P/E or
   P/B at least 15% below the stock's own trailing 3-year average")
   requires EPS/book-value/shares-outstanding history that isn't in the
   Phase 1 fundamentals schema (`revenue, pat, roe, roce, debt_to_equity,
   promoter_holding, pledged_pct, operating_cash_flow`). Rather than fake
   this rule as always-true, `signals/beaten_down_screen.py` marks it
   explicitly unverifiable and the screen never fires. To fix: extend the
   fundamentals CSV schema with EPS and book value per share, and implement
   the trailing-3yr-average comparison in that file's `rule6` section.

3. **Backtester deviates from the plan.** The plan suggested `vectorbt` or
   `backtrader`. This build uses a custom, dependency-light event-driven
   backtester (`backtest/engine.py`) instead, because the exit logic
   (target/stop/trailing/time, evaluated day-by-day per open position) is
   path-dependent and doesn't map cleanly onto vectorbt's vectorized
   signal-array model. If you specifically want vectorbt/backtrader, this
   would need to be rewritten.

4. **Survivorship bias is not solved.** The backtester only has access to
   the *current* universe list, so stocks that were delisted/merged/renamed
   during the backtest window are silently excluded. This makes reported
   metrics optimistic versus what a true historical run would show. Per
   the plan, source a historical point-in-time constituent list before
   trusting these numbers for real capital decisions.

5. **Delivery % does not come from Dhan.** Dhan's historical daily-candle
   endpoint returns only OHLCV — no delivery quantity. Delivery % is
   pulled separately from NSE bhavcopy via the `jugaad-data` library
   (handles NSE's pre/post July-2024 format change). NSE scraping is
   inherently a bit flaky; ingestion logs a warning and continues with
   null delivery data rather than failing the whole run if it's
   unavailable for a stretch of dates.

6. **Nifty 50 index data must be ingested separately** under the symbol
   `NIFTY50` for the breakout screen's relative-strength rule (#5) to work.
   Nothing does this automatically yet — see "Ingest Nifty 50" below.

7. **Nothing here has been run against a real Dhan account.** No API
   credentials were available during development. Every module was
   verified with synthetic OHLCV data (config loading, storage round-trip,
   swing detection, both screens' rule evaluation, the backtest engine's
   trade simulation, and the paper-trade daily runner all ran correctly on
   synthetic input). The Dhan API usage itself (`dhanhq` SDK calls, the
   scrip-master CSV URL, response schema, and the 20 req/sec non-trading
   rate limit) was confirmed against Dhan's own published documentation and
   GitHub repo, but has not been exercised against live credentials.

## Setup

```powershell
cd trading-agent
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
Copy-Item .env.example .env
# then edit .env: DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN, DHAN_API_KEY
```

`pandas-ta` is commented out in `requirements.txt` — its pinned version
isn't installable on recent Python/numpy combos. RSI falls back to a
built-in Wilder's RSI implementation automatically if `pandas_ta` isn't
present, so this is safe to leave as-is.

## Running the pipeline, in order

```powershell
# One-time: seed the NSE holiday calendar (2024-2026, compiled from
# published exchange calendars — extend data_pipeline/universe/nse_holidays_seed.csv
# for future years and re-run this)
python -m data_pipeline.ingest_holidays

# Pull OHLCV + delivery % for the universe (needs .env credentials)
python -m data_pipeline.ingest_ohlcv --years 2

# Ingest Nifty 50 index data (required for the breakout screen's
# relative-strength rule) — override --symbols with the index's own
# security_id/trading symbol as listed in Dhan's scrip master (index
# segment, not equity)
python -m data_pipeline.ingest_ohlcv --years 2 --symbols NIFTY50

# Quarterly: ingest fundamentals from a hand-compiled CSV
# (see data_pipeline/universe/fundamentals_sample.csv for the exact shape)
python -m data_pipeline.ingest_fundamentals path\to\fundamentals.csv

# Phase 2: run today's screens, writes data/shortlists/<date>.csv
python -m signals.run_screens

# Phase 3: the gate — backtest both screens vs a random baseline
python -m backtest.run_backtest --years 2
# writes backtest_report.md and backtest_trades.csv to the project root

# Phase 4: one day of paper trading (run once per trading day, e.g. via
# Windows Task Scheduler after market close)
python -m paper_trade.daily_run --ingest
```

Only proceed past Phase 3 if `backtest_report.md` shows the combined
screens beating the random baseline on the primary metric (% of flagged
stocks hitting +20% within 7 trading days) — the report explicitly calls
this out if it doesn't.

## Project layout

```
trading-agent/
├── config.py                  # every tunable in one place — no hardcoding elsewhere
├── data_pipeline/
│   ├── dhan_client.py          # Dhan SDK wrapper + rate limiting
│   ├── instrument_master.py    # symbol -> Dhan security_id mapping
│   ├── storage.py              # DataProvider (Parquet-backed)
│   ├── quality_checks.py       # Phase 1 acceptance-criteria checks
│   ├── ingest_ohlcv.py
│   ├── ingest_fundamentals.py
│   ├── ingest_holidays.py
│   └── universe/                # universe list, peer groups, holiday seed, fundamentals sample
├── signals/
│   ├── swing_detector.py       # fractal/zigzag swings, trend structure, ATH regime
│   ├── breakout_screen.py
│   ├── beaten_down_screen.py
│   └── run_screens.py          # daily shortlist runner
├── backtest/
│   ├── engine.py                # trade simulation, next-day-open execution
│   ├── metrics.py                # win rate, Sharpe/Sortino, primary metric, etc.
│   ├── random_baseline.py
│   └── run_backtest.py
├── paper_trade/
│   └── daily_run.py
└── data/, logs/                 # created at runtime, gitignored
```

## Sources consulted

- Dhan Python SDK usage, historical data method signature and response
  schema: [dhan-oss/DhanHQ-py](https://github.com/dhan-oss/DhanHQ-py),
  [Historical Data — DhanHQ v2 docs](https://dhanhq.co/docs/v2/historical-data/)
- Dhan API rate limits: [Dhan support article](https://dhan.freshdesk.com/support/solutions/articles/82000891163-what-are-the-api-limits-per-second-in-dhan-)
- Dhan scrip master CSV: [Instrument List — DhanHQ v2 docs](https://dhanhq.co/docs/v2/instruments/)
- NSE bhavcopy / delivery data format and library: [jugaad-data historical data guide](https://github.com/jugaad-py/jugaad-data/blob/master/docs/HISTORICAL_DATA_GUIDE.md)
- NSE holiday calendars (2024-2026): [Zerodha holiday calendar](https://zerodha.com/marketintel/holiday-calendar/), [Aditya Birla Capital 2025 list](https://www.adityabirlacapital.com/abc-of-money/share-market-holidays-2025), [Economic Times 2024 list](https://www.indiatimes.com/investment/stock-market-holiday-list-2024-bse-nse-to-remain-shut-on-these-dates/articleshow/126680556.html)
