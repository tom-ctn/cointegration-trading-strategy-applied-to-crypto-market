import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta, timezone

# ==========================
# CONFIG
# ==========================

# Put your pairs here (can be 10, 50, 400...)
PAIRS = [
    ("APTUSDT", "ARBUSDT"),
    ("APTUSDT", "ATOMUSDT"),
    ("APTUSDT", "OPUSDT"),
    ("NEARUSDT", "OPUSDT"),
    # Cointegration candidates (p < 0.05)
    ("FTMUSDT", "HIVEUSDT"),
    ("AKTUSDT", "KSMUSDT"),
    ("AKTUSDT", "DOTUSDT"),
    ("HIVEUSDT", "WAXPUSDT"),
    ("AKTUSDT", "ONEUSDT"),
    ("AKTUSDT", "VETUSDT"),
    ("AKTUSDT", "EGLDUSDT"),
    ("KSMUSDT", "MINAUSDT"),
    ("AKTUSDT", "HIVEUSDT"),
    ("AKTUSDT", "FLOWUSDT"),
    ("XEMUSDT", "ZECUSDT"),
    ("APTUSDT", "DOTUSDT"),
    ("APTUSDT", "KSMUSDT"),
    ("XEMUSDT", "XLMUSDT"),
    ("KSMUSDT", "ROSEUSDT"),
    ("AKTUSDT", "ZILUSDT"),
    ("XEMUSDT", "ZILUSDT"),
    ("AKTUSDT", "CELOUSDT"),
    ("XEMUSDT", "XTZUSDT"),
    ("ASTRUSDT", "HIVEUSDT"),
    ("AKTUSDT", "FILUSDT"),
    ("FILUSDT", "HIVEUSDT"),
    ("CKBUSDT", "DOTUSDT"),
    ("CKBUSDT", "KSMUSDT"),
    ("EGLDUSDT", "HIVEUSDT"),
    ("HIVEUSDT", "ZILUSDT"),
    ("APTUSDT", "HIVEUSDT"),
    ("DOTUSDT", "MINAUSDT"),
    ("ASTRUSDT", "KSMUSDT"),
    ("CKBUSDT", "XTZUSDT"),
    ("ASTRUSDT", "DOTUSDT"),
    ("AKTUSDT", "WAXPUSDT"),
    ("DOTUSDT", "ROSEUSDT"),
    ("HIVEUSDT", "ONTUSDT"),
    ("CELOUSDT", "DOTUSDT"),
    ("CELOUSDT", "HIVEUSDT"),
    ("EGLDUSDT", "KSMUSDT"),
    ("FLOWUSDT", "HIVEUSDT"),
    ("AKTUSDT", "ALGOUSDT"),
    ("CELOUSDT", "KSMUSDT"),
    ("KSMUSDT", "NEARUSDT"),
    ("XEMUSDT", "XMRUSDT"),
    ("HIVEUSDT", "VETUSDT"),
    ("AKTUSDT", "ATOMUSDT"),
    ("FILUSDT", "MINAUSDT"),
    ("DOTUSDT", "HIVEUSDT"),
    ("HIVEUSDT", "ONEUSDT"),
    ("MINAUSDT", "ONEUSDT"),
    ("DOTUSDT", "FLOWUSDT"),
    ("AKTUSDT", "ONTUSDT"),
    ("APTUSDT", "EGLDUSDT"),
    ("FILUSDT", "KSMUSDT"),
    ("ATOMUSDT", "RUNEUSDT"),
    ("APTUSDT", "ZILUSDT"),
    ("KSMUSDT", "ZILUSDT"),
    ("APTUSDT", "ATOMUSDT"),  # duplicated, fine
    ("MINAUSDT", "VETUSDT"),
    ("APTUSDT", "CELOUSDT"),
    ("APTUSDT", "ONEUSDT"),
    ("KSMUSDT", "OPUSDT"),
    ("CELOUSDT", "MINAUSDT"),
    ("MINAUSDT", "ZILUSDT"),
    ("CELOUSDT", "CKBUSDT"),
    ("AKTUSDT", "ROSEUSDT"),
    ("HIVEUSDT", "MINAUSDT"),
    ("EGLDUSDT", "MINAUSDT"),
    ("ASTRUSDT", "EGLDUSDT"),
    ("APTUSDT", "FLOWUSDT"),
    ("ASTRUSDT", "ONEUSDT"),
    ("DOTUSDT", "ZILUSDT"),
    ("CKBUSDT", "HIVEUSDT"),
    ("ASTRUSDT", "VETUSDT"),
    ("KSMUSDT", "WAXPUSDT"),
    ("CKBUSDT", "ONEUSDT"),
    ("FLOWUSDT", "KSMUSDT"),
    ("ALGOUSDT", "ETCUSDT"),
    ("APTUSDT", "ASTRUSDT"),
    ("APTUSDT", "VETUSDT"),
    ("MINAUSDT", "OPUSDT"),
    ("HIVEUSDT", "KSMUSDT"),
    ("CELOUSDT", "OPUSDT"),
    ("DOTUSDT", "NEARUSDT"),
    ("APTUSDT", "FILUSDT"),
    ("CKBUSDT", "VETUSDT"),
    ("CKBUSDT", "FLOWUSDT"),
    ("FLOWUSDT", "MINAUSDT"),
    ("ROSEUSDT", "VETUSDT"),
    ("APTUSDT", "WAXPUSDT"),
    ("AKTUSDT", "ASTRUSDT"),
    ("DOTUSDT", "OPUSDT"),
    ("KSMUSDT", "RUNEUSDT"),
    ("OPUSDT", "ZILUSDT"),
    ("ONTUSDT", "WAXPUSDT"),
    ("KSMUSDT", "ONEUSDT"),
    ("ASTRUSDT", "ZILUSDT"),
    ("NEARUSDT", "VETUSDT"),
    ("ATOMUSDT", "ZILUSDT"),
    ("ICPUSDT", "MINAUSDT"),
    ("ADAUSDT", "ETCUSDT"),
    ("DOTUSDT", "EGLDUSDT"),
    ("CFXUSDT", "XLMUSDT"),
    ("ADAUSDT", "INJUSDT"),
    ("AKTUSDT", "CKBUSDT"),
    ("VETUSDT", "WAXPUSDT"),
    ("XEMUSDT", "XRPUSDT"),
    ("APTUSDT", "TIAUSDT"),
    ("HIVEUSDT", "OPUSDT"),
    ("FILUSDT", "OPUSDT"),
    ("ADAUSDT", "ARBUSDT"),
    ("APTUSDT", "ROSEUSDT"),
    ("ONEUSDT", "ROSEUSDT"),
    ("ATOMUSDT", "HIVEUSDT"),
    ("OPUSDT", "ROSEUSDT"),
    ("ALGOUSDT", "CKBUSDT"),
    ("CKBUSDT", "OPUSDT"),
    ("CKBUSDT", "ZILUSDT"),
    ("EGLDUSDT", "ONEUSDT"),
    ("FILUSDT", "ONEUSDT"),
    ("AKTUSDT", "MINAUSDT"),
    ("ALGOUSDT", "INJUSDT"),
    ("ASTRUSDT", "FILUSDT"),
    ("FLOWUSDT", "OPUSDT"),
    ("NEARUSDT", "XLMUSDT"),
    ("CELOUSDT", "ROSEUSDT"),
    ("HIVEUSDT", "ROSEUSDT"),
    ("MINAUSDT", "XTZUSDT"),
    ("INJUSDT", "KSMUSDT"),
    ("ALGOUSDT", "ROSEUSDT"),
    ("ALGOUSDT", "FLOWUSDT"),
    ("ALGOUSDT", "NEARUSDT"),
    ("ALGOUSDT", "NEOUSDT"),
    ("ATOMUSDT", "CKBUSDT"),
    ("EGLDUSDT", "OPUSDT"),
    ("DOTUSDT", "KSMUSDT"),
    ("ALGOUSDT", "ARBUSDT"),
    ("NEARUSDT", "ONEUSDT"),
    ("CKBUSDT", "ETCUSDT"),
    ("ALGOUSDT", "ONTUSDT"),
    ("FTMUSDT", "RUNEUSDT"),
    ("DOTUSDT", "RUNEUSDT"),
    ("ALGOUSDT", "WAXPUSDT"),
    ("EGLDUSDT", "VETUSDT"),
    ("ALGOUSDT", "ASTRUSDT"),
    ("OPUSDT", "VETUSDT"),
    ("XLMUSDT", "XTZUSDT"),
    ("KSMUSDT", "NEOUSDT"),
    ("CELOUSDT", "ONEUSDT"),
    ("KSMUSDT", "ONTUSDT"),
    ("ALGOUSDT", "MINAUSDT"),
    ("ADAUSDT", "SEIUSDT"),
    ("RUNEUSDT", "XTZUSDT"),
    ("HIVEUSDT", "TIAUSDT"),
    ("ALGOUSDT", "OPUSDT"),
    ("SEIUSDT", "XLMUSDT"),
    ("ICPUSDT", "ROSEUSDT"),
    ("ALGOUSDT", "XTZUSDT"),
    ("APTUSDT", "MINAUSDT"),
    ("ALGOUSDT", "ZILUSDT"),
    ("ICXUSDT", "INJUSDT"),
    ("FILUSDT", "ROSEUSDT"),
    ("CKBUSDT", "ICPUSDT"),
    ("CKBUSDT", "FILUSDT"),
    ("ADAUSDT", "CKBUSDT"),
    ("ADAUSDT", "AKTUSDT"),
    ("ALGOUSDT", "ONEUSDT"),
    ("RUNEUSDT", "VETUSDT"),
    ("APTUSDT", "ONTUSDT"),
    ("LTCUSDT", "XLMUSDT"),
    ("ONEUSDT", "RUNEUSDT"),
    ("ALGOUSDT", "QTUMUSDT"),
    ("ADAUSDT", "AVAXUSDT"),
    ("SOLUSDT", "XLMUSDT"),
    ("ARBUSDT", "XTZUSDT"),
    ("ONTUSDT", "VETUSDT"),
    ("ONEUSDT", "WAXPUSDT"),
    ("AKTUSDT", "NEOUSDT"),
    ("ONEUSDT", "OPUSDT"),
    ("OPUSDT", "XTZUSDT"),
    ("ICXUSDT", "RVNUSDT"),
    ("EGLDUSDT", "WAXPUSDT"),
    ("ASTRUSDT", "CELOUSDT"),
    ("ADAUSDT", "QTUMUSDT"),
    ("ASTRUSDT", "ICPUSDT"),
    ("ASTRUSDT", "OPUSDT"),
    ("ALGOUSDT", "CELOUSDT"),
    ("ICPUSDT", "KSMUSDT"),
    ("ALGOUSDT", "APTUSDT"),
    ("INJUSDT", "ONEUSDT"),
    ("ATOMUSDT", "MINAUSDT"),
    ("CKBUSDT", "EGLDUSDT"),
    ("INJUSDT", "VETUSDT"),
    ("ATOMUSDT", "INJUSDT"),
    ("ASTRUSDT", "FLOWUSDT"),
    ("CELOUSDT", "VETUSDT"),
    ("ICPUSDT", "OPUSDT"),
    ("FTMUSDT", "NEARUSDT"),
    ("FLOWUSDT", "ONEUSDT"),
    ("NEOUSDT", "OPUSDT"),
    ("AKTUSDT", "HBARUSDT"),
    ("FTMUSDT", "SEIUSDT"),
    ("NEOUSDT", "VETUSDT"),
    ("ATOMUSDT", "CELOUSDT"),
    ("ADAUSDT", "NEARUSDT"),
    ("ALGOUSDT", "EGLDUSDT"),
    ("ALGOUSDT", "AVAXUSDT"),
    ("EGLDUSDT", "FILUSDT"),
    ("RUNEUSDT", "ZILUSDT"),
    ("ONEUSDT", "ZILUSDT"),
    ("AKTUSDT", "XTZUSDT"),
    ("INJUSDT", "XLMUSDT"),
    ("KSMUSDT", "VETUSDT"),
    ("AKTUSDT", "ICPUSDT"),
    ("ALGOUSDT", "ICXUSDT"),
    ("HBARUSDT", "SUIUSDT"),
    ("FILUSDT", "ZILUSDT"),
    ("ALGOUSDT", "VETUSDT"),
    ("ALGOUSDT", "ATOMUSDT"),
    ("ALGOUSDT", "ICPUSDT"),
    ("AKTUSDT", "QTUMUSDT"),
    ("AKTUSDT", "OPUSDT"),
    ("APTUSDT", "ICPUSDT"),
    ("ADAUSDT", "OPUSDT"),
    ("TIAUSDT", "ZILUSDT"),
    ("APTUSDT", "FTMUSDT"),
    ("APTUSDT", "CKBUSDT"),
    ("CELOUSDT", "ZILUSDT"),
    ("ETCUSDT", "XLMUSDT"),
    ("ASTRUSDT", "ONTUSDT"),
    ("ALGOUSDT", "KSMUSDT"),
    ("CELOUSDT", "NEARUSDT"),
    ("ADAUSDT", "APTUSDT"),
    ("FTMUSDT", "ROSEUSDT"),
    ("HBARUSDT", "XLMUSDT"),
    ("ADAUSDT", "ALGOUSDT"),
    ("ASTRUSDT", "ATOMUSDT"),
    ("ICPUSDT", "NEARUSDT"),
    ("ALGOUSDT", "FILUSDT"),
    ("LTCUSDT", "SOLUSDT"),
    ("ETCUSDT", "INJUSDT"),
    ("FILUSDT", "WAXPUSDT"),
    ("XRPUSDT", "ZILUSDT"),
    ("ALGOUSDT", "DOTUSDT"),
    ("DOTUSDT", "ONEUSDT"),
    ("APTUSDT", "NEARUSDT"),
    ("FTMUSDT", "RVNUSDT"),
    ("HBARUSDT", "INJUSDT"),
    ("AKTUSDT", "TIAUSDT"),
    ("MINAUSDT", "TIAUSDT"),
    ("XRPUSDT", "ZECUSDT"),
    ("DOTUSDT", "WAXPUSDT"),
    ("ONTUSDT", "OPUSDT"),
    ("ONEUSDT", "ONTUSDT"),
    ("FILUSDT", "NEARUSDT"),
    ("FTMUSDT", "MINAUSDT"),
    ("ALGOUSDT", "RUNEUSDT"),
    ("APTUSDT", "QTUMUSDT"),
    ("ADAUSDT", "SUIUSDT"),
    ("FILUSDT", "VETUSDT"),
    ("CKBUSDT", "ROSEUSDT"),
    ("NEARUSDT", "ROSEUSDT"),
    ("XRPUSDT", "XTZUSDT"),
    ("VETUSDT", "ZILUSDT"),
    ("ADAUSDT", "XTZUSDT"),
    ("ATOMUSDT", "ROSEUSDT"),
    ("APTUSDT", "ICXUSDT"),
    ("ONEUSDT", "VETUSDT"),
    ("ADAUSDT", "ONTUSDT"),
    ("DOTUSDT", "INJUSDT"),
    ("ICPUSDT", "ONEUSDT"),
    ("CELOUSUT", "TIAUSDT"),  # typo, will likely be skipped
    ("CKBUSDT", "NEARUSDT"),
    ("CKBUSDT", "HBARUSDT"),
    ("NEARUSDT", "NEOUSDT"),
    ("AKTUSDT", "FTMUSDT"),
    ("CKBUSDT", "QTUMUSDT"),
    ("MINAUSDT", "ROSEUSDT"),
    ("ADAUSDT", "MINAUSDT"),
    ("ADAUSDT", "XMRUSDT"),
    ("CELOUSDT", "RUNEUSDT"),
    ("ALGOUSDT", "HIVEUSDT"),
    ("CELOUSDT", "FLOWUSDT"),
    ("NEOUSDT", "XTZUSDT"),
    ("AKTUSDT", "APTUSDT"),
    ("EGLDUSDT", "TIAUSDT"),
    ("HBARUSDT", "NEOUSDT"),
    ("ICPUSDT", "XTZUSDT"),
    ("ADAUSDT", "RUNEUSDT"),
    ("ADAUSDT", "WAXPUSDT"),
    ("HBARUSDT", "SOLUSDT"),
    ("ALGOUSDT", "SEIUSDT"),
    ("NEARUSDT", "XTZUSDT"),
    ("FLOWUSDT", "VETUSDT"),
    ("CELOUSDT", "XTZUSDT"),
    ("ASTRUSDT", "MINAUSDT"),
    ("HIVEUSDT", "NEOUSDT"),
    ("ADAUSDT", "NEOUSDT"),
    ("ADAUSDT", "FLOWUSDT"),
    ("ADAUSDT", "ICXUSDT"),
    ("AKTUSDT", "RVNUSDT"),
    ("HBARUSDT", "NEARUSDT"),
    ("APTUSDT", "NEOUSDT"),
    ("FLOWUSDT", "NEARUSDT"),
    ("RVNUSDT", "SEIUSDT"),
    ("ALGOUSDT", "RVNUSDT"),
    ("CELOUSDT", "ICPUSDT"),
    ("ASTRUSDT", "ROSEUSDT"),
    ("ADAUSDT", "ASTRUSDT"),
    ("ATOMUSDT", "ONEUSDT"),
    ("HBARUSDT", "OPUSDT"),
    ("ATOMUSDT", "WAXPUSPT"),  # typo, will be skipped
    ("ADAUSDT", "ZILUSDT"),
    ("HIVEUSDT", "RUNEUSDT"),
    ("CKBUSDT", "FTMUSDT"),
    ("ADAUSDT", "CELOUSDT"),
    ("ADAUSDT", "ROSEUSDT"),
    ("FTMUSDT", "INJUSDT"),
    ("DOTUSDT", "FILUSDT"),
    ("FLOWUSDT", "ROSEUSDT"),
    ("INJUSDT", "QTUMUSDT"),
    ("LTCUSDT", "SEIUSDT"),
    ("FTMUSDT", "QTUMUSDT"),
    ("HBARUSDT", "ROSEUSDT"),
    ("CKBUSDT", "ONTUSDT"),
    ("HBARUSDT", "ONTUSDT"),
    ("HBARUSDT", "SEIUSDT"),
    ("FLOWUSDT", "RUNEUSDT"),
    ("CKBUSDT", "NEOUSDT"),
    ("HBARUSDT", "WAXPUSDT"),
    ("HBARUSDT", "MINAUSDT"),
    ("AKTUSDT", "ICXUSDT"),
    ("ETCUSDT", "HIVEUSDT"),
    ("TIAUSDT", "VETUSDT"),
    ("WAXPUSDT", "ZILUSDT"),
    ("HBARUSDT", "QTUMUSDT"),
    ("ETCUSDT", "ICXUSDT"),
    ("INJUSDT", "RVNUSDT"),
    ("ASTRUSDT", "TIAUSDT"),
    ("MINAUSDT", "NEOUSDT"),
    ("OPUSDT", "WAXPUSDT"),
    ("FILUSDT", "TIAUSDT"),
    ("ADAUSDT", "KSMUSDT"),
    ("XLMUSDT", "ZILUSDT"),
    ("LTCUSDT", "XMRUSDT"),
    ("ARBUSDT", "ETCUSDT"),
    ("FILUSDT", "ICPUSDT"),
    ("EOSUSDT", "SEIUSDT"),
    ("ALGOUSDT", "TIAUSDT"),
    ("ARBUSDT", "HIVEUSDT"),
    ("HBARUSDT", "XTZUSDT"),
    ("MINAUSDT", "XLMUSDT"),
    ("CELOUSDT", "EGLDUSDT"),
    ("ARBUSDT", "CKBUSDT"),
    ("LTCUSDT", "MINAUSDT"),
    ("LTCUSDT", "NEARUSDT"),
    ("NEARUSDT", "OPUSDT"),
    ("LTCUSDT", "NEOUSDT"),
    ("ROSEUSDT", "TIAUSDT"),
    ("FTMUSDT", "OPUSDT"),
    ("LTCUSDT", "XTZUSDT"),
    ("LTCUSDT", "RUNEUSDT"),
    ("DOTUSDT", "ONTUSDT"),
    ("LTCUSDT", "OPUSDT"),
    ("FTMUSDT", "ZILUSDT"),
    ("NEARUSDT", "ZILUSDT"),
    ("CKBUSDT", "XLMUSDT"),
    ("DOTUSDT", "TIAUSDT"),
    ("FILUSDT", "QTUMUSDT"),
    ("LTCUSDT", "ROSEUSDT"),
    ("ADAUSDT", "EGLDUSDT"),
    ("FTMUSDT", "ONTUSDT"),
    ("KSMUSDT", "RVNUSDT"),
    ("FTMUSDT", "XTZUSDT"),
    ("CELOUSDT", "WAXPUSDT"),
    ("ADAUSDT", "TIAUSDT"),
    ("FILUSDT", "RUNEUSDT"),
    ("ARBUSDT", "INJUSDT"),
    ("ASTRUSDT", "WAXPUSDT"),
    ("KSMUSDT", "XTZUSDT"),
    ("AKTUSDT", "INJUSDT"),
    ("LTCUSDT", "WAXPUSDT"),
    ("QTUMUSDT", "XTZUSDT"),
    ("OPUSDT", "QTUMUSDT"),
    ("ONTUSDT", "TIAUSDT"),
    ("HBARUSDT", "ZILUSDT"),
    ("NEARUSDT", "QTUMUSDT"),
    ("FTMUSDT", "ICPUSDT"),
    ("HBARUSDT", "RUNEUSDT"),
    ("MINAUSDT", "WAXPUSDT"),
    ("CELOUSDT", "FILUSDT"),
    ("LTCUSDT", "ONTUSDT"),
    ("EOSUSDT", "ZECUSDT"),
    ("LTCUSDT", "ZECUSDT"),
    ("EOSUSDT", "RUNEUSDT"),
    ("ADAUSDT", "CFXUSDT"),
    ("FLOWUSDT", "ICPUSDT"),
    ("HBARUSDT", "ICPUSDT"),
    ("LTCUSDT", "XRPUSDT"),
    ("FTMUSDT", "TIAUSDT"),
    ("LTCUSDT", "RVNUSDT"),
    ("ADAUSDT", "XLMUSDT"),
    ("LTCUSDT", "TIAUSDT"),
    ("FTMUSDT", "NEOUSDT"),
    ("MINAUSDT", "QTUMUSDT"),
]

INTERVAL = "4h"
LOOKBACK_DAYS = 730          # total history to fetch

# Z-score strategy params (same for all pairs)
ENTRY_Z = 2.0
EXIT_Z  = 0.5
STOP_Z  = 4.0
MAX_HOLD_BARS = 200          # None to disable
FEE_RATE = 0.0004            # round-trip fee per leg (we apply 2*FEE_RATE per trade)

# Rolling cointegration parameters
ROLLING_WINDOW_DAYS = 90     # rolling window length
PVAL_THRESHOLD = 0.05        # only trade when p-value < this
MAX_HALF_LIFE_DAYS = 30      # only trade when half-life <= this (set None to disable)

# Portfolio construction
N_TOP_SHARPE = 20            # keep best N pairs by Sharpe ratio
TARGET_ANNUAL_VOL = 0.20     # target portfolio volatility (before leverage cap)
MAX_LEVERAGE = 5.0           # max leverage applied to risk-parity portfolio

PLOT_RESULTS = True


# ==========================
# DATA FETCHING
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

    if len(df) < 200:
        raise ValueError(f"Not enough overlapping data for pair {symbol_x}-{symbol_y}")

    return df


# ==========================
# ROLLING COINTEGRATION
# ==========================

def get_periods_per_day(interval: str) -> float:
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
        return 1.0


def estimate_half_life(resid, freq_per_day):
    """Estimate mean-reversion half-life in days from residual series."""
    resid = pd.Series(resid).dropna()
    if len(resid) < 10:
        return np.nan

    lagged = resid.shift(1).dropna()
    delta = resid.diff().dropna()
    delta = delta.loc[lagged.index]
    X = sm.add_constant(lagged.values)
    model = sm.OLS(delta.values, X).fit()
    phi = model.params[1]

    if phi >= 0:
        return np.nan
    hl_bars = -np.log(2) / phi
    return hl_bars / freq_per_day


def compute_rolling_cointegration(prices, rolling_window_days):
    """
    For a price dataframe with columns X, Y, compute rolling:
      - beta
      - spread
      - z-score
      - p-value (ADF on residuals)
      - half-life
    """
    df = prices.copy()
    df["beta_roll"] = np.nan
    df["spread_roll"] = np.nan
    df["z_roll"] = np.nan
    df["pval_roll"] = np.nan
    df["halflife_days"] = np.nan

    periods_per_day = get_periods_per_day(INTERVAL)
    window_bars = int(rolling_window_days * periods_per_day)
    if window_bars < 20:
        raise ValueError("Rolling window too short in bars.")

    print(f"  Rolling CI: window = {rolling_window_days} days "
          f"({window_bars} bars)")

    for i in range(window_bars, len(df)):
        window = df.iloc[i - window_bars:i]
        Xw = window["X"]
        Yw = window["Y"]

        # Regression X ~ Y
        Y_mat = sm.add_constant(Yw)
        model = sm.OLS(Xw, Y_mat).fit()
        beta = model.params["Y"]
        resid = model.resid

        # ADF test on residuals -> p-value for stationarity
        try:
            adf_res = adfuller(resid, maxlag=1, autolag="AIC")
            pval = adf_res[1]
        except Exception:
            pval = np.nan

        # Half-life of mean reversion
        hl_days = estimate_half_life(resid, periods_per_day)

        # spread & z-score at time t
        x_t = df["X"].iloc[i]
        y_t = df["Y"].iloc[i]
        spread_series = Xw - beta * Yw
        mu = spread_series.mean()
        sigma = spread_series.std()
        spread_t = x_t - beta * y_t
        z_t = (spread_t - mu) / sigma if sigma > 0 else np.nan

        idx = df.index[i]
        df.at[idx, "beta_roll"] = beta
        df.at[idx, "spread_roll"] = spread_t
        df.at[idx, "z_roll"] = z_t
        df.at[idx, "pval_roll"] = pval
        df.at[idx, "halflife_days"] = hl_days

    return df


# ==========================
# BACKTEST (PAIR LEVEL)
# ==========================

def backtest_zscore_pair(df,
                         entry_z,
                         exit_z,
                         stop_z,
                         max_hold_bars=None,
                         pval_threshold=None,
                         max_halflife_days=None):
    """
    Z-score mean-reversion backtest on one pair using rolling CI columns:
      - uses spread_roll, z_roll, beta_roll
      - trades only when pval_roll < pval_threshold and halflife <= max_halflife_days
    Returns trades dataframe and equity series (start = 1.0).
    """
    equity = 1.0
    equity_series = pd.Series(index=df.index, dtype=float)
    trades = []

    position = 0
    entry_idx = None
    entry_spread = None
    entry_notional = None
    equity_entry = None
    bars_in_trade = 0

    for i, (ts, row) in enumerate(df.iterrows()):
        z = row["z_roll"]
        s = row["spread_roll"]
        x = row["X"]
        y = row["Y"]
        beta = row["beta_roll"]

        # If any key field missing, just carry equity
        if pd.isna(z) or pd.isna(s) or pd.isna(beta):
            equity_series[ts] = equity
            continue

        # Filters based on p-value and half-life
        if pval_threshold is not None:
            pval = row["pval_roll"]
            if pd.notna(pval) and pval >= pval_threshold:
                # CI not valid -> flatten if needed
                if position != 0:
                    equity_series[ts] = equity
                    position = 0
                    entry_idx = entry_spread = entry_notional = equity_entry = None
                    bars_in_trade = 0
                else:
                    equity_series[ts] = equity
                continue

        if max_halflife_days is not None:
            hl = row["halflife_days"]
            if pd.notna(hl) and hl > max_halflife_days:
                if position != 0:
                    equity_series[ts] = equity
                    position = 0
                    entry_idx = entry_spread = entry_notional = equity_entry = None
                    bars_in_trade = 0
                else:
                    equity_series[ts] = equity
                continue

        # FLAT -> look for entry
        if position == 0:
            equity_series[ts] = equity
            if z > entry_z:
                # short spread
                position = -1
                entry_idx = ts
                entry_spread = s
                entry_notional = abs(x) + abs(beta * y)
                equity_entry = equity
                bars_in_trade = 0
            elif z < -entry_z:
                # long spread
                position = 1
                entry_idx = ts
                entry_spread = s
                entry_notional = abs(x) + abs(beta * y)
                equity_entry = equity
                bars_in_trade = 0
            continue

        # IN TRADE -> mark to market
        bars_in_trade += 1
        mtm_ret = position * (s - entry_spread) / entry_notional
        equity_mtm = equity_entry * (1.0 + mtm_ret)
        equity_series[ts] = equity_mtm

        exit_reason = None

        if abs(z) < exit_z:
            exit_reason = "MeanRevert"
        if abs(z) > stop_z:
            exit_reason = "StopZ"
        if max_hold_bars is not None and bars_in_trade >= max_hold_bars:
            exit_reason = "MaxHold"
        if i == len(df) - 1 and exit_reason is None:
            exit_reason = "EoD"

        if exit_reason is not None:
            exit_ts = ts
            spread_pnl = position * (s - entry_spread) / entry_notional
            fee_pct = -2 * FEE_RATE
            trade_ret = spread_pnl + fee_pct

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
    if interval.endswith("m"):
        minutes = int(interval[:-1])
        per_day = (24 * 60) / minutes
    elif interval.endswith("h"):
        hours = int(interval[:-1])
        per_day = 24 / hours
    elif interval.endswith("d"):
        days = int(interval[:-1])
        per_day = 1 / days
    else:
        per_day = 1.0
    return per_day * 365.0


def compute_metrics(equity_series: pd.Series, interval: str):
    equity = equity_series.dropna()
    if len(equity) < 10:
        return None

    rets = equity.pct_change().dropna()
    if rets.empty or rets.std() == 0:
        return None

    ann_factor = get_annual_factor(interval)
    mean_ret = rets.mean()
    std_ret = rets.std()

    sharpe = (mean_ret / std_ret) * np.sqrt(ann_factor)

    downside = rets[rets < 0]
    sortino = None
    if not downside.empty and downside.std() > 0:
        sortino = (mean_ret / downside.std()) * np.sqrt(ann_factor)

    start_eq = equity.iloc[0]
    end_eq = equity.iloc[-1]
    total_ret = end_eq / start_eq - 1
    periods_per_year = ann_factor
    years = len(rets) / periods_per_year if periods_per_year > 0 else 0
    if years > 0:
        cagr = (end_eq / start_eq) ** (1 / years) - 1
    else:
        cagr = None
    dd = (equity / equity.cummax() - 1).min()
    calmar = cagr / abs(dd) if (cagr is not None and dd < 0) else None

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "total_return": total_ret,
        "cagr": cagr,
        "max_dd": dd,
        "final_equity": end_eq,
    }


def print_metrics(label, metrics):
    if metrics is None:
        print(f"\n{label}: insufficient data.")
        return
    print(f"\n=== {label} ===")
    print(f"Sharpe       : {metrics['sharpe']:.3f}")
    print(f"Sortino      : {metrics['sortino']:.3f}" if metrics["sortino"] is not None else "Sortino      : n/a")
    print(f"Calmar       : {metrics['calmar']:.3f}" if metrics["calmar"] is not None else "Calmar       : n/a")
    print(f"Total return : {metrics['total_return']:.2%}")
    print(f"CAGR         : {metrics['cagr']:.2%}" if metrics["cagr"] is not None else "CAGR         : n/a")
    print(f"Max drawdown : {metrics['max_dd']:.2%}")
    print(f"Final equity : {metrics['final_equity']:.3f}")


# ==========================
# PORTFOLIO BACKTEST
# ==========================

def backtest_portfolio(pairs):
    per_pair_equity = {}
    per_pair_metrics = {}
    all_trades = []
    all_index = set()

    # 1) Backtest every pair
    for sym_x, sym_y in pairs:
        pair_name = f"{sym_x}-{sym_y}"
        print(f"\n===== Processing pair {pair_name} =====")
        try:
            prices = load_pair_prices(sym_x, sym_y, INTERVAL, LOOKBACK_DAYS)
            df_roll = compute_rolling_cointegration(prices, ROLLING_WINDOW_DAYS)
            trades_df, equity_series = backtest_zscore_pair(
                df_roll,
                entry_z=ENTRY_Z,
                exit_z=EXIT_Z,
                stop_z=STOP_Z,
                max_hold_bars=MAX_HOLD_BARS,
                pval_threshold=PVAL_THRESHOLD,
                max_halflife_days=MAX_HALF_LIFE_DAYS,
            )

            metrics = compute_metrics(equity_series, INTERVAL)
            print_metrics(f"Pair {pair_name}", metrics)

            if metrics is None:
                print(f"[WARN] No valid metrics for {pair_name}, skipping.")
                continue

            per_pair_equity[pair_name] = equity_series
            per_pair_metrics[pair_name] = metrics
            all_index.update(equity_series.index)

            if not trades_df.empty:
                trades_df["pair"] = pair_name
                all_trades.append(trades_df)

        except Exception as e:
            print(f"[ERROR] Skipping {pair_name}: {e}")

    if not per_pair_equity:
        raise ValueError("No valid pairs for portfolio.")

    # 2) Select top N pairs by Sharpe
    metrics_df = pd.DataFrame(per_pair_metrics).T  # index: pair_name
    metrics_df = metrics_df.dropna(subset=["sharpe"])
    metrics_df = metrics_df.sort_values("sharpe", ascending=False)

    top_pairs = metrics_df.head(N_TOP_SHARPE).index.tolist()
    print(f"\nUsing top {len(top_pairs)} pairs by Sharpe:")
    for name in top_pairs:
        print(f"  {name}: Sharpe = {metrics_df.loc[name, 'sharpe']:.3f}")

    # Filter to top pairs only
    per_pair_equity_top = {name: per_pair_equity[name] for name in top_pairs}
    per_pair_metrics_top = {name: per_pair_metrics[name] for name in top_pairs}

    # 3) Build aligned equity dataframe for top pairs
    all_index = sorted(set().union(*[s.index for s in per_pair_equity_top.values()]))
    eq_df = pd.DataFrame(index=all_index)
    for pair_name, eq_series in per_pair_equity_top.items():
        eq_df[pair_name] = eq_series.reindex(all_index).ffill()

    # 4) Risk-parity weights (1/vol) based on unlevered returns
    rets_df = eq_df.pct_change().dropna()
    ann_factor = get_annual_factor(INTERVAL)
    vols_ann = rets_df.std() * np.sqrt(ann_factor)
    inv_vol = 1.0 / vols_ann
    weights = inv_vol / inv_vol.sum()

    print("\nPair weights (risk parity, before leverage):")
    for name, w in weights.items():
        print(f"{name:25s} w = {w:.4f}")

    # Unlevered portfolio returns & equity
    port_rets_unlev = (rets_df * weights).sum(axis=1)
    eq_unlev = (1 + port_rets_unlev).cumprod()
    eq_unlev = eq_unlev.reindex(eq_df.index).ffill().fillna(1.0)

    # Unlevered annual vol and leverage
    unlev_vol_ann = port_rets_unlev.std() * np.sqrt(ann_factor)
    if unlev_vol_ann <= 0:
        leverage = 1.0
    else:
        leverage = min(TARGET_ANNUAL_VOL / unlev_vol_ann, MAX_LEVERAGE)

    print(f"\nUnlevered annual vol: {unlev_vol_ann:.2%}")
    print(f"Target annual vol   : {TARGET_ANNUAL_VOL:.2%}")
    print(f"Applied leverage    : {leverage:.2f}x")

    # Levered portfolio equity
    port_rets_lev = port_rets_unlev * leverage
    eq_lev = (1 + port_rets_lev).cumprod()
    eq_lev = eq_lev.reindex(eq_df.index).ffill().fillna(1.0)

    portfolio_metrics = compute_metrics(eq_lev, INTERVAL)

    if all_trades:
        trades_all = pd.concat(all_trades, ignore_index=True)
    else:
        trades_all = pd.DataFrame()

    return per_pair_equity_top, per_pair_metrics_top, eq_lev, portfolio_metrics, trades_all


# ==========================
# PLOTTING
# ==========================

def plot_portfolio(per_pair_equity, portfolio_equity):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Per-pair equity curves
    ax1 = axes[0]
    for name, eq in per_pair_equity.items():
        ax1.plot(eq.index, eq, label=name)
    ax1.set_title("Per-pair equity curves (rolling CI strategy, top N Sharpe)")
    ax1.set_ylabel("Equity")
    ax1.legend(ncol=2, fontsize=8)

    # Portfolio equity
    ax2 = axes[1]
    eqp = portfolio_equity.dropna()
    ax2.plot(eqp.index, eqp, label="Portfolio Equity (risk parity, levered)")
    ax2.axhline(1.0, color="black", linestyle="--", alpha=0.7)
    ax2.set_title("Portfolio P&L (multi-pair rolling CI, top N Sharpe)")
    ax2.set_ylabel("Equity")
    ax2.set_xlabel("Date")
    ax2.legend()

    plt.tight_layout()
    plt.show()


# ==========================
# MAIN
# ==========================

def main():
    per_pair_equity, per_pair_metrics, portfolio_equity, portfolio_metrics, trades_all = backtest_portfolio(PAIRS)

    print_metrics("PORTFOLIO", portfolio_metrics)

    print(f"\nTotal number of trades (all pairs): {len(trades_all)}")
    if not trades_all.empty:
        print("\nSample trades:")
        print(trades_all.head(15).to_string(index=False))

    if PLOT_RESULTS:
        plot_portfolio(per_pair_equity, portfolio_equity)


if __name__ == "__main__":
    main()
