# خوارزمية تداول (Trading Algorithm) — Maaden (Tadawul: 1211)

A daily-timeframe trading algorithm for **Saudi Arabian Mining Co. (Maaden, 1211.SR)**, backtested against Yahoo Finance data. Implemented in both Python ([`maaden_strategy.py`](maaden_strategy.py)) and C++ ([`maaden_strategy.cpp`](maaden_strategy.cpp)), which are cross-checked against each other to produce bit-identical results.

## Strategy

Long-only **SMA crossover with an RSI entry filter and a trailing stop**:

- **Entry**: 10-day SMA crosses above the 30-day SMA, and RSI(14) is below 85 (avoids buying into a blow-off top).
- **Exit**: 10-day SMA crosses back below the 30-day SMA (trend reversal), or price falls 8% from its peak since entry (trailing stop — cuts losers fast, lets winners run).
- Long-only, since Tadawul has no retail short-selling. 0.1% commission modeled per trade.

## A signal-generation bug was found and fixed (important)

An earlier version of `maaden_strategy.py` computed the crossover entry signal as:

```python
cross_up = trend_up & ~trend_up.shift(1).fillna(False)
```

`trend_up.shift(1)` introduces a leading `NaN`, which silently upcasts the boolean Series to `object` dtype. Python's `~` on an `object`-dtype `True`/`False` does **bitwise** negation (`~True == -2`, `~False == -1`), not logical negation — so `cross_up` fired on ~43% of all days instead of only on genuine crossovers (roughly 2%). This was caught by cross-checking against the independently-written C++ port, which used plain `bool` logic and never hit this pandas dtype trap; the two implementations disagreed until the Python bug was fixed.

**Fix**: `trend_up.shift(1, fill_value=False)` preserves `bool` dtype and avoids the trap. Both implementations now produce identical output for the same input data.

This means earlier performance figures generated during development (multi-hundred-combination parameter searches across several rule variants) were computed with the over-firing signal and are no longer trustworthy. Only the numbers below, generated after the fix, should be relied on.

## Verified results (post-fix)

| Scenario | Strategy | Buy & hold |
|---|---|---|
| 100,000 SAR lump sum, 2018 → 2026-09 | +138.7% (44 trades, Sharpe 0.59, max DD −45.9%) | +258.4% |
| 990 SAR lump sum, 2025 → 2026-09 | **+27.5%** (9 trades) | +22.7% — **strategy wins** |
| 990 SAR lump sum, 2024 → 2026-09 | +23.4% (14 trades) | +23.7% — essentially tied |
| 990 SAR/month DCA, 2024 → 2026-09 (33 contributions, 32,670 SAR invested) | +17.0% (14 trades, 35.7% win rate) | +17.3% — essentially tied |

Over the full 2018-2026 history the strategy still trails buy-and-hold by a wide margin (Maaden's long-run trend has been too clean and low-drawdown for signal timing to beat just holding). But over the more recent windows, the corrected strategy is much more competitive than the pre-fix numbers suggested — it even edges ahead over 2025-2026. A backtest describes the past; it is not a promise about the future.

## Usage

### Python

```bash
pip install -r requirements.txt

# Full backtest + current buy/hold/sell stance
python maaden_strategy.py
```

For lump-sum or monthly-DCA scenarios with custom capital/dates, import the module functions directly:

```python
import maaden_strategy as m

# Lump sum
df = m.fetch_data(m.TICKER, "2024-01-01")
df = m.generate_signals(m.add_indicators(df))
result = m.backtest(df)  # uses m.INITIAL_CAPITAL

# Monthly DCA
m.dca_scenario(m.TICKER, "2024-01-01", monthly_amount=990)
```

### C++

The C++ port reads price data from a CSV (`Date,Close`) rather than fetching it directly — export one from Python first, then build and run:

```bash
python -c "
import maaden_strategy as m
df = m.fetch_data(m.TICKER, '2024-01-01')
out = df.reset_index()[['Date','Close']]
out['Date'] = out['Date'].dt.strftime('%Y-%m-%d')
out.to_csv('data_1211SR.csv', index=False)
"

g++ -O2 -std=c++17 -static -static-libgcc -static-libstdc++ -o maaden_strategy maaden_strategy.cpp
./maaden_strategy data_1211SR.csv 990
```

(`-static -static-libgcc -static-libstdc++` avoids crashes from mismatched `libstdc++` DLLs if more than one MinGW toolchain is on `PATH`, e.g. Git for Windows bundles its own.)

`data_1211SR.csv` (2024-01-01 onward) is included in this repo so the C++ version can be run without Python installed.

## Disclaimer

This is a backtesting/research tool, not financial advice. Past performance does not guarantee future results. No execution/broker integration is included — it only generates signals and historical performance stats.
