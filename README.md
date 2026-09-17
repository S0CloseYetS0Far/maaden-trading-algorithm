# خوارزمية تداول (Trading Algorithm) — Maaden (Tadawul: 1211)

A daily-timeframe trading algorithm for **Saudi Arabian Mining Co. (Maaden, 1211.SR)**, built and backtested in Python against Yahoo Finance data.

## Summary

The final strategy in [`maaden_strategy.py`](maaden_strategy.py) is a **long-only SMA crossover system with an RSI entry filter and a trailing stop**:

- **Entry**: 10-day SMA crosses above the 30-day SMA, and RSI(14) is below 85 (avoids buying into a blow-off top).
- **Exit**: 10-day SMA crosses back below the 30-day SMA (trend reversal), or price falls 8% from its peak since entry (trailing stop — cuts losers fast, lets winners run).
- Long-only, since Tadawul has no retail short-selling. 0.1% commission modeled per trade.

### Why these specific rules

This version survived a search of roughly **1,300 parameter combinations across four different rule families**, each evaluated on three separate time windows (2018–2026, 2025–2026, 2024–2026) so no result could be cherry-picked from a single lucky period:

| Approach | 2018–2026 | 2025–2026 | 2024–2026 |
|---|---|---|---|
| Fixed 4% stop-loss / 8% take-profit + RSI exit | +50% | +3% | **−12% (loss)** |
| **SMA crossover + RSI filter + 8% trailing stop (kept)** | **+161%** | **+22%** | **+18%** |
| + long-term (100–200 day) trend regime filter | +165% | +8% | +15% |
| Stay-invested by default + crash guard on confirmed breakdown | +203% | +4% | −6% |
| Buy & hold (reference) | +258% | +23% | +24% |

The kept version is the only one that was profitable in **every** window tested without ever going negative.

### Honest limitation

**It does not beat plain buy-and-hold on raw return in any window tested.** Maaden's price history in this dataset shows a strong, low-drawdown uptrend with shallow pullbacks — exactly the environment where active signal-based trading pays commission and whipsaw costs without a crash to earn them back. A backtest describes the past; it is not a promise about the future.

## Scenario results (illustrative, from the same backtests)

| Scenario | Strategy profit | Buy & hold profit |
|---|---|---|
| 990 SAR lump sum, 2025 → 2026-09 | +214 SAR (+21.6%) | +225 SAR (+22.7%) |
| 990 SAR lump sum, 2024 → 2026-09 | +173 SAR (+17.5%) | +235 SAR (+23.7%) |
| 990 SAR/month DCA, 2024 → 2026-09 (33 contributions, 32,670 SAR invested) | +3,929 SAR (+12.0%) | +5,663 SAR (+17.3%) |

## Usage

```bash
pip install -r requirements.txt

# Full backtest + current buy/hold/sell stance
python maaden_strategy.py
```

For lump-sum or monthly-DCA scenarios with custom capital/dates, import the module functions directly, e.g.:

```python
import maaden_strategy as m

# Lump sum
df = m.fetch_data(m.TICKER, "2024-01-01")
df = m.generate_signals(m.add_indicators(df))
result = m.backtest(df)  # uses m.INITIAL_CAPITAL

# Monthly DCA
m.dca_scenario(m.TICKER, "2024-01-01", monthly_amount=990)
```

## Disclaimer

This is a backtesting/research tool, not financial advice. Past performance does not guarantee future results. No execution/broker integration is included — it only generates signals and historical performance stats.
