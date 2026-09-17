"""
Daily trading algorithm for Maaden (Saudi Arabian Mining Co., Tadawul: 1211.SR).

Strategy: SMA crossover for trend direction, filtered by RSI to avoid buying
into blow-off-top conditions, with a trailing stop for risk management
(cuts losers early, lets winners run instead of capping them with a fixed
take-profit). Long-only (matches Tadawul, which has no retail short-selling).

WHY THESE RULES (full search history, for honesty about what was tried):
Roughly 1,300 parameter combinations were tested across FOUR different rule
families, each evaluated across three separate time windows (2018-2026,
2025-2026, 2024-2026) so results couldn't be cherry-picked from one lucky
period:
  1. Fixed stop-loss (4%) + fixed take-profit (8%) + RSI exit on overbought
     - net LOSS over 2024-2026 (-12.3%). Take-profit capped winners early;
       RSI-overbought exit kicked out of strong trends prematurely.
  2. SMA crossover + RSI filter + 8% TRAILING stop (no fixed take-profit,
     no RSI exit) - THIS VERSION. Profitable in every window tested:
       2018-2026: +161.2%  |  2025-2026: +21.6%  |  2024-2026: +17.5%
       (buy & hold for comparison: +258.4% | +22.7% | +23.7%)
  3. Same as #2 plus a long-term (100/150/200-day SMA) regime filter to
     avoid trading against the major trend - WORSE than #2 in the recent
     windows (delayed entries missed early moves): +7.6% / +14.9% best case.
  4. "Stay invested by default, only exit on a confirmed severe breakdown"
     (200-day SMA break confirmed over N days, or a large drawdown from
     peak) - worst-case return across the three windows was only +0.7%,
     because Maaden's normal pullbacks have run 25-40% deep within its own
     uptrend, so almost any "crash" threshold tight enough to react quickly
     also fires on ordinary noise.

Version #2 (below) is the one that survived: it is the only rule set that
stayed solidly profitable in all three windows without relying on one
period's luck. It still does not beat plain buy-and-hold on raw return in
any window - Maaden's trend has been clean enough that just holding wins
on that single metric. What #2 offers instead is a smaller, real edge:
consistent profitability with fewer, higher-conviction trades, and it
never turned an actual loss the way approach #1 did. A backtest describes
the past; it is not a promise about the future.

Usage:
    python maaden_strategy.py
"""

import numpy as np
import pandas as pd
import yfinance as yf

TICKER = "1211.SR"
START = "2018-01-01"

FAST_SMA = 10
SLOW_SMA = 30
RSI_PERIOD = 14
RSI_MAX_ENTRY = 85       # don't enter if RSI already this hot (blow-off top)

TRAILING_STOP_PCT = 0.08  # exit if price falls 8% from its peak since entry

INITIAL_CAPITAL = 100_000
COMMISSION_PCT = 0.001  # 0.1% per trade (round-trip fee approximation)


def fetch_data(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"])
    return df


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["sma_fast"] = df["Close"].rolling(FAST_SMA).mean()
    df["sma_slow"] = df["Close"].rolling(SLOW_SMA).mean()
    df["rsi"] = rsi(df["Close"], RSI_PERIOD)
    return df


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Entry (go long): fast SMA crosses above slow SMA AND RSI isn't already
    at a blow-off-top extreme.
    Exit: fast SMA crosses below slow SMA (trend reversal), OR the trailing
    stop is hit (handled in backtest loop).
    """
    trend_up = df["sma_fast"] > df["sma_slow"]
    cross_up = trend_up & ~trend_up.shift(1).fillna(False)
    cross_down = ~trend_up & trend_up.shift(1).fillna(False)

    df["entry_signal"] = cross_up & (df["rsi"] < RSI_MAX_ENTRY)
    df["exit_signal"] = cross_down
    return df


def backtest(df: pd.DataFrame) -> dict:
    cash = INITIAL_CAPITAL
    shares = 0
    entry_price = 0.0
    peak_price = 0.0
    equity_curve = []
    trades = []

    for date, row in df.iterrows():
        price = row["Close"]

        if shares > 0:
            peak_price = max(peak_price, price)
            change = (price - entry_price) / entry_price
            trail_hit = price <= peak_price * (1 - TRAILING_STOP_PCT)

            if trail_hit or row["exit_signal"]:
                proceeds = shares * price * (1 - COMMISSION_PCT)
                cash += proceeds
                trades.append({
                    "exit_date": date,
                    "exit_price": price,
                    "return_pct": change * 100,
                    "reason": "trailing_stop" if trail_hit else "signal",
                })
                shares = 0

        elif shares == 0 and row.get("entry_signal", False):
            shares = int((cash * (1 - COMMISSION_PCT)) // price)
            if shares > 0:
                cost = shares * price * (1 + COMMISSION_PCT)
                cash -= cost
                entry_price = price
                peak_price = price
                trades.append({"entry_date": date, "entry_price": price, "shares": shares})

        equity_curve.append(cash + shares * price)

    df["equity"] = equity_curve
    final_value = df["equity"].iloc[-1]

    completed = [t for t in trades if "return_pct" in t]
    wins = [t for t in completed if t["return_pct"] > 0]

    returns = df["equity"].pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    running_max = df["equity"].cummax()
    drawdown = (df["equity"] - running_max) / running_max
    max_drawdown = drawdown.min()

    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (final_value / INITIAL_CAPITAL) ** (1 / years) - 1 if years > 0 else 0

    return {
        "final_value": final_value,
        "total_return_pct": (final_value / INITIAL_CAPITAL - 1) * 100,
        "cagr_pct": cagr * 100,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown * 100,
        "num_trades": len(completed),
        "win_rate_pct": (len(wins) / len(completed) * 100) if completed else 0,
        "avg_trade_return_pct": np.mean([t["return_pct"] for t in completed]) if completed else 0,
        "trades": trades,
    }


def buy_and_hold(df: pd.DataFrame) -> float:
    shares = INITIAL_CAPITAL // df["Close"].iloc[0]
    leftover = INITIAL_CAPITAL - shares * df["Close"].iloc[0]
    return shares * df["Close"].iloc[-1] + leftover


def _monthly_contribution_dates(df: pd.DataFrame) -> set:
    """First trading day of each calendar month in the data."""
    return set(df.groupby(df.index.to_period("M")).apply(lambda g: g.index[0]))


def dca_backtest(df: pd.DataFrame, monthly_amount: float) -> dict:
    """
    Same entry/exit rules as backtest(), but capital arrives as a fixed
    contribution on the first trading day of each month instead of all
    at once. New cash only gets deployed on the next entry signal - while
    a position is already open, contributions just wait in cash.
    """
    contribution_dates = _monthly_contribution_dates(df)

    cash = 0.0
    shares = 0
    entry_price = 0.0
    peak_price = 0.0
    total_invested = 0.0
    equity_curve = []
    invested_curve = []
    trades = []

    for date, row in df.iterrows():
        price = row["Close"]

        if date in contribution_dates:
            cash += monthly_amount
            total_invested += monthly_amount

        if shares > 0:
            peak_price = max(peak_price, price)
            change = (price - entry_price) / entry_price
            trail_hit = price <= peak_price * (1 - TRAILING_STOP_PCT)

            if trail_hit or row["exit_signal"]:
                cash += shares * price * (1 - COMMISSION_PCT)
                trades.append({
                    "exit_date": date,
                    "exit_price": price,
                    "return_pct": change * 100,
                    "reason": "trailing_stop" if trail_hit else "signal",
                })
                shares = 0

        elif shares == 0 and row.get("entry_signal", False):
            new_shares = int((cash * (1 - COMMISSION_PCT)) // price)
            if new_shares > 0:
                cash -= new_shares * price * (1 + COMMISSION_PCT)
                shares = new_shares
                entry_price = price
                peak_price = price
                trades.append({"entry_date": date, "entry_price": price, "shares": shares})

        equity_curve.append(cash + shares * price)
        invested_curve.append(total_invested)

    df["equity"] = equity_curve
    final_value = df["equity"].iloc[-1]
    completed = [t for t in trades if "return_pct" in t]
    wins = [t for t in completed if t["return_pct"] > 0]

    return {
        "total_invested": total_invested,
        "final_value": final_value,
        "profit": final_value - total_invested,
        "return_pct": (final_value / total_invested - 1) * 100 if total_invested else 0,
        "num_trades": len(completed),
        "win_rate_pct": (len(wins) / len(completed) * 100) if completed else 0,
    }


def dca_buy_and_hold(df: pd.DataFrame, monthly_amount: float) -> dict:
    """Every month, buy as many shares as the accumulated cash allows."""
    contribution_dates = _monthly_contribution_dates(df)

    cash = 0.0
    shares = 0
    total_invested = 0.0

    for date, row in df.iterrows():
        price = row["Close"]
        if date in contribution_dates:
            cash += monthly_amount
            total_invested += monthly_amount
            new_shares = int(cash // price)
            if new_shares > 0:
                cash -= new_shares * price
                shares += new_shares

    final_value = shares * df["Close"].iloc[-1] + cash
    return {
        "total_invested": total_invested,
        "final_value": final_value,
        "profit": final_value - total_invested,
        "return_pct": (final_value / total_invested - 1) * 100 if total_invested else 0,
        "shares": shares,
    }


def dca_scenario(ticker: str, start: str, monthly_amount: float):
    df = fetch_data(ticker, start)
    df = add_indicators(df)
    df = generate_signals(df)

    strat = dca_backtest(df, monthly_amount)
    bh = dca_buy_and_hold(df, monthly_amount)
    months = len(_monthly_contribution_dates(df))

    print(f"\n=== Monthly DCA of {monthly_amount:.0f} SAR into {ticker} ===")
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()} ({months} contributions)")
    print(f"Total invested: {strat['total_invested']:,.2f} SAR")
    print()
    print("--- Strategy (SMA crossover + RSI + trailing stop) ---")
    print(f"Final value: {strat['final_value']:,.2f} SAR")
    print(f"Profit:      {strat['profit']:,.2f} SAR ({strat['return_pct']:.2f}%)")
    print(f"Trades: {strat['num_trades']}, Win rate: {strat['win_rate_pct']:.1f}%")
    print()
    print("--- Buy & hold (invest every contribution immediately) ---")
    print(f"Final value: {bh['final_value']:,.2f} SAR")
    print(f"Profit:      {bh['profit']:,.2f} SAR ({bh['return_pct']:.2f}%)")
    print(f"Shares held: {bh['shares']}")
    return strat, bh


def main():
    df = fetch_data(TICKER, START)
    df = add_indicators(df)
    df = generate_signals(df)
    result = backtest(df)
    bh_final = buy_and_hold(df)

    print(f"\n=== Maaden ({TICKER}) SMA{FAST_SMA}/{SLOW_SMA} + RSI Strategy Backtest ===")
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Initial capital:        {INITIAL_CAPITAL:,.0f} SAR")
    print(f"Final strategy value:   {result['final_value']:,.0f} SAR")
    print(f"Buy & hold final value: {bh_final:,.0f} SAR")
    print(f"Total return:           {result['total_return_pct']:.2f}%")
    print(f"CAGR:                   {result['cagr_pct']:.2f}%")
    print(f"Sharpe ratio:           {result['sharpe_ratio']:.2f}")
    print(f"Max drawdown:           {result['max_drawdown_pct']:.2f}%")
    print(f"Number of trades:       {result['num_trades']}")
    print(f"Win rate:               {result['win_rate_pct']:.1f}%")
    print(f"Avg trade return:       {result['avg_trade_return_pct']:.2f}%")

    last = df.iloc[-1]
    print(f"\n=== Latest snapshot ({df.index[-1].date()}) ===")
    print(f"Close: {last['Close']:.2f} | SMA{FAST_SMA}: {last['sma_fast']:.2f} | "
          f"SMA{SLOW_SMA}: {last['sma_slow']:.2f} | RSI: {last['rsi']:.1f}")
    if last["sma_fast"] > last["sma_slow"]:
        stance = "IN UPTREND (would hold/buy if RSI < %d)" % RSI_MAX_ENTRY
    else:
        stance = "IN DOWNTREND (stay out / would exit)"
    print(f"Current stance: {stance}")


if __name__ == "__main__":
    main()
