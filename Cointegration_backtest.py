import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from datetime import datetime, timedelta, timezone

# ==========================
# CONFIG
# ==========================

SYMBOL_X = "APTUSDT"     # X leg
SYMBOL_Y = "ONEUSDT"       # Y leg

INTERVAL = "4h"
LOOKBACK_DAYS = 730

ENTRY_Z = 2.0            # enter when |z| > ENTRY_Z
EXIT_Z  = 0.5            # exit when |z| < EXIT_Z
STOP_Z  = 4.0            # emergency stop if |z| > STOP_Z
MAX_HOLD_BARS = 200      # None to disable

FEE_RATE = 0.0004        # per notional, both legs combined we charge 2*FEE_RATE per trade

# Rolling beta window
ROLLING_WINDOW_DAYS = 90   # length of window used for rolling beta

PLOT_RESULTS = True


# ==========================
# DATA
# ==========================

def binance_futures_klines(symbol, interval, start_time_ms, end_time_ms):
    base_url = "https://fapi.binance.com/fapi/v1/klines"
    all_rows = []
    start = start_time_ms
    limit = 1500

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start,
            "endTime": end_time_ms,
            "limit": limit
        }
        r = requests.get(base_url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            break

        all_rows.extend(data)
        last_open_time = data[-1][0]
        start = last_open_time + 1
        if len(data) < limit:
            break

    if not all_rows:
        raise ValueError(f"No kline data for {symbol}")

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(all_rows, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    df["close"] = df["close"].astype(float)
    return df[["close"]]


def load_pair_prices(symbol_x, symbol_y, interval, lookback_days):
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=lookback_days)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    print(f"Fetching {symbol_x}...")
    df_x = binance_futures_klines(symbol_x, interval, start_ms, end_ms)
    print(f"Fetching {symbol_y}...")
    df_y = binance_futures_klines(symbol_y, interval, start_ms, end_ms)

    df = pd.concat(
        [df_x.rename(columns={"close": "X"}),
         df_y.rename(columns={"close": "Y"})],
        axis=1
    ).dropna()

    if len(df) < 100:
        raise ValueError("Not enough overlapping data for the pair.")

    return df


# ==========================
# HELPER: periods per day
# ==========================

def get_periods_per_day(interval: str) -> float:
    """Convert Binance interval string to approximate bars per day."""
    if interval.endswith("m"):
        minutes = int(interval[:-1])
        return (24 * 60) / minutes
    elif interval.endswith("h"):
        hours = int(interval[:-1])
        return 24 / hours
    elif interval.endswith("d"):
        days = int(interval[:-1])
        return 1 / days
    else:
        # fallback: treat as daily
        return 1.0


# ==========================
# COINTEGRATION & SPREAD (with rolling beta)
# ==========================

def compute_spread_and_zscore(prices):
    # ----- Static cointegration & beta (for information only) -----
    stat, pval, crit = coint(prices["X"], prices["Y"])
    print("\n=== Cointegration test (static, full sample) ===")
    print(f"Test statistic: {stat:.4f}")
    print(f"P-value      : {pval:.6f}")
    print("Critical vals:", dict(zip(["1%", "5%", "10%"], crit)))
    if pval < 0.05:
        print("=> Likely cointegrated at 5% level.")
    else:
        print("=> Weak CI evidence; trade with caution.")

    Y_full = sm.add_constant(prices["Y"])
    model_full = sm.OLS(prices["X"], Y_full).fit()
    alpha_full = model_full.params["const"]
    beta_full = model_full.params["Y"]
    print("\n=== Static hedge ratio (X ~ Y, full sample) ===")
    print(f"alpha: {alpha_full:.4f}")
    print(f"beta : {beta_full:.4f}")

    # ----- Rolling beta over a window in days -----
    df = prices.copy()
    df["beta"] = np.nan

    periods_per_day = get_periods_per_day(INTERVAL)
    window_bars = int(ROLLING_WINDOW_DAYS * periods_per_day)
    if window_bars < 20:
        raise ValueError("Rolling window too short in bars.")

    print(f"\nRolling beta: window = {ROLLING_WINDOW_DAYS} days "
          f"~ {window_bars} bars")

    for i in range(window_bars, len(df)):
        window_x = df["X"].iloc[i - window_bars:i]
        window_y = df["Y"].iloc[i - window_bars:i]

        Y_win = sm.add_constant(window_y)
        model_win = sm.OLS(window_x, Y_win).fit()
        beta_t = model_win.params["Y"]

        df.iloc[i, df.columns.get_loc("beta")] = beta_t

    # Spread using rolling beta
    df["spread"] = df["X"] - df["beta"] * df["Y"]

    # z-score based on full-sample spread stats (same style as before)
    spread_valid = df["spread"].dropna()
    mu = spread_valid.mean()
    sigma = spread_valid.std()

    df["zscore"] = (df["spread"] - mu) / sigma

    return df, mu, sigma


# ==========================
# BACKTEST (with equity path)
# ==========================

def backtest_pairs_strategy(df,
                            entry_z,
                            exit_z,
                            stop_z,
                            max_hold_bars=None):
    """
    Uses df['beta'] (rolling) and df['zscore'].
    Returns:
      trades_df       : per-trade statistics
      equity_series   : pandas Series, equity path over all timestamps
    Equity starts at 1.0 and evolves mark-to-market while in a trade.
    """
    trades = []
    equity = 1.0
    equity_series = pd.Series(index=df.index, dtype=float)

    position = 0            # 0 = flat, 1 = long spread, -1 = short spread
    entry_idx = None
    entry_spread = None
    entry_notional = None
    equity_entry = None     # equity just before entering trade
    bars_in_trade = 0

    for i, (ts, row) in enumerate(df.iterrows()):
        z = row["zscore"]
        s = row["spread"]
        x = row["X"]
        y = row["Y"]
        beta_t = row["beta"]

        # if z or beta not available -> stay flat / carry equity
        if np.isnan(z) or np.isnan(beta_t):
            equity_series[ts] = equity
            continue

        # FLAT -> look for entry
        if position == 0:
            equity_series[ts] = equity

            if z > entry_z:
                # SHORT spread: -X + beta*Y
                position = -1
                entry_idx = ts
                entry_spread = s
                entry_notional = abs(x) + abs(beta_t * y)
                equity_entry = equity
                bars_in_trade = 0

            elif z < -entry_z:
                # LONG spread: +X - beta*Y
                position = 1
                entry_idx = ts
                entry_spread = s
                entry_notional = abs(x) + abs(beta_t * y)
                equity_entry = equity
                bars_in_trade = 0

            continue

        # IN TRADE -> mark-to-market
        bars_in_trade += 1
        mtm_ret = position * (s - entry_spread) / entry_notional
        equity_mtm = equity_entry * (1.0 + mtm_ret)

        equity_series[ts] = equity_mtm
        exit_reason = None

        # mean-reversion exit
        if abs(z) < exit_z:
            exit_reason = "MeanRevert"

        # hard stop
        if abs(z) > stop_z:
            exit_reason = "StopZ"

        # max holding
        if max_hold_bars is not None and bars_in_trade >= max_hold_bars:
            exit_reason = "MaxHold"

        # last bar -> close anyway
        if i == len(df) - 1 and exit_reason is None:
            exit_reason = "EOD"

        if exit_reason is not None:
            exit_ts = ts
            exit_spread = s

            # realized PnL in spread units
            spread_pnl = position * (exit_spread - entry_spread) / entry_notional

            # apply fees once, as a percent of notional
            fee_pct = -2 * FEE_RATE
            trade_ret = spread_pnl + fee_pct

            # update equity to realized level
            equity = equity_entry * (1.0 + trade_ret)
            equity_series[ts] = equity

            trades.append({
                "entry_time": entry_idx,
                "exit_time": exit_ts,
                "direction": "LONG_SPREAD" if position == 1 else "SHORT_SPREAD",
                "bars_held": bars_in_trade,
                "exit_reason": exit_reason,
                "return_pct": trade_ret,
            })

            # reset
            position = 0
            entry_idx = entry_spread = entry_notional = equity_entry = None
            bars_in_trade = 0

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["cum_return"] = (1 + trades_df["return_pct"]).cumprod() - 1

    return trades_df, equity_series


# ==========================
# PERFORMANCE METRICS
# ==========================

def get_annual_factor(interval: str) -> float:
    """
    Approximate number of periods per year given a Binance interval string.
    Supports 'Xm', 'Xh', 'Xd'.
    """
    if interval.endswith("m"):
        minutes = int(interval[:-1])
        periods_per_day = (24 * 60) / minutes
        return periods_per_day * 365
    elif interval.endswith("h"):
        hours = int(interval[:-1])
        periods_per_day = 24 / hours
        return periods_per_day * 365
    elif interval.endswith("d"):
        days = int(interval[:-1])
        return 365 / days
    else:
        # fallback: assume daily
        return 365.0


def compute_ratios(equity_series: pd.Series, interval: str):
    """
    Compute Sharpe, Sortino, Calmar from the equity curve.
    Assumes equity_series is continuous over bars and starts at >0.
    Risk-free rate assumed 0.
    """
    equity = equity_series.dropna()
    if len(equity) < 10:
        return None, None, None

    # per-period returns
    rets = equity.pct_change().dropna()
    if rets.empty or rets.std() == 0:
        return None, None, None

    ann_factor = get_annual_factor(interval)

    mean_ret = rets.mean()
    std_ret = rets.std()

    # Sharpe
    sharpe = (mean_ret / std_ret) * np.sqrt(ann_factor)

    # Sortino
    downside = rets[rets < 0]
    if downside.empty or downside.std() == 0:
        sortino = None
    else:
        sortino = (mean_ret / downside.std()) * np.sqrt(ann_factor)

    # Calmar: CAGR / |max drawdown|
    start_eq = equity.iloc[0]
    end_eq = equity.iloc[-1]
    periods_per_year = ann_factor
    years = len(rets) / periods_per_year if periods_per_year > 0 else 0
    if years <= 0:
        calmar = None
    else:
        cagr = (end_eq / start_eq) ** (1 / years) - 1
        dd = (equity / equity.cummax() - 1).min()  # negative
        calmar = cagr / abs(dd) if dd < 0 else None

    return sharpe, sortino, calmar


# ==========================
# REPORTING / PLOTS
# ==========================

def print_summary(trades_df, equity_series):
    equity = equity_series.dropna()

    if trades_df.empty:
        print("\nNo trades generated with current parameters.")
    else:
        print("\n=== TRADES ===")
        print(trades_df[[
            "entry_time", "exit_time", "direction",
            "bars_held", "exit_reason", "return_pct"
        ]].to_string(index=False))

        print("\n=== SUMMARY (per-trade) ===")
        n = len(trades_df)
        win_rate = (trades_df["return_pct"] > 0).mean()
        avg_ret = trades_df["return_pct"].mean()
        total_ret = trades_df["cum_return"].iloc[-1]
        max_dd = (equity.div(equity.cummax()) - 1).min()

        print(f"Number of trades : {n}")
        print(f"Win rate         : {win_rate:.2%}")
        print(f"Avg return/trade : {avg_ret:.4%}")
        print(f"Total return     : {total_ret:.2%}")
        print(f"Max drawdown     : {max_dd:.2%}")

    # Performance ratios from equity path
    sharpe, sortino, calmar = compute_ratios(equity_series, INTERVAL)
    print("\n=== Performance Ratios (from equity curve, rf=0) ===")
    print(f"Sharpe  : {sharpe:.3f}" if sharpe is not None else "Sharpe  : n/a")
    print(f"Sortino : {sortino:.3f}" if sortino is not None else "Sortino : n/a")
    print(f"Calmar  : {calmar:.3f}" if calmar is not None else "Calmar  : n/a")

    if not equity.empty:
        print("\nFinal equity:", equity.iloc[-1])


def plot_results(df, trades_df, equity_series):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 1) Normalized prices
    axes[0].plot(df.index, df["X"] / df["X"].iloc[0], label=SYMBOL_X)
    axes[0].plot(df.index, df["Y"] / df["Y"].iloc[0], label=SYMBOL_Y)
    axes[0].set_title(f"{SYMBOL_X} vs {SYMBOL_Y} (normalized futures prices)")
    axes[0].set_ylabel("Normalized Price")
    axes[0].legend()

    # 2) Spread + entry thresholds (unchanged style)
    axes[1].plot(df.index, df["spread"], label="Spread")
    spread_valid = df["spread"].dropna()
    mu = spread_valid.mean()
    sigma = spread_valid.std()
    axes[1].axhline(mu, color="black", linestyle="--", label="Mean")
    axes[1].axhline(mu + ENTRY_Z * sigma, color="red", linestyle="--", alpha=0.7, label="±entry")
    axes[1].axhline(mu - ENTRY_Z * sigma, color="red", linestyle="--", alpha=0.7)
    axes[1].set_ylabel("Spread")

    if not trades_df.empty:
        for _, tr in trades_df.iterrows():
            idx = tr["entry_time"]
            axes[1].axvline(
                idx,
                color="green" if tr["direction"] == "LONG_SPREAD" else "orange",
                linestyle=":", alpha=0.5
            )

    axes[1].legend()

    # 3) Equity curve (P&L evolution)
    eq = equity_series.dropna()
    axes[2].plot(eq.index, eq, label="Equity (MTM)")
    axes[2].axhline(1.0, color="black", linestyle="--", alpha=0.7)
    axes[2].set_ylabel("Equity")
    axes[2].set_xlabel("Date")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


# ==========================
# MAIN
# ==========================

def main():
    prices = load_pair_prices(SYMBOL_X, SYMBOL_Y, INTERVAL, LOOKBACK_DAYS)
    df, mu, sigma = compute_spread_and_zscore(prices)
    trades_df, equity_series = backtest_pairs_strategy(
        df,
        ENTRY_Z,
        EXIT_Z,
        STOP_Z,
        max_hold_bars=MAX_HOLD_BARS
    )
    print_summary(trades_df, equity_series)

    if PLOT_RESULTS:
        plot_results(df, trades_df, equity_series)


if __name__ == "__main__":
    main()
