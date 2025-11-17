import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from itertools import combinations
from statsmodels.tsa.stattools import coint

# ==========================
# CONFIG
# ==========================

# L1 tokens to test (Binance USDT futures symbols)
L1_TOKENS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "APTUSDT",
    "AVAXUSDT",
    "NEARUSDT",
    "ATOMUSDT",
    "ARBUSDT",
    "OPUSDT",
    "ADAUSDT",
    "ALGOUSDT",
    "ASTRUSDT",
    "BCHUSDT",
    "BNBUSDT",
    "CANTOUSDT",
    "CELOUSDT",
    "CFXUSDT",
    "CKBUSDT",
    "CROUSDT",
    "DOTUSDT",
    "EGLDUSDT",
    "EOSUSDT",
    "ETCUSDT",
    "FILUSDT",
    "FLOWUSDT",
    "FTMUSDT",
    "HBARUSDT",
    "ICPUSDT",
    "ICXUSDT",
    "INJUSDT",
    "KAVAUSDT",
    "KLAYUSDT",
    "KSMUSDT",
    "LTCUSDT",
    "MINAUSDT",
    "NEOUSDT",
    "ONEUSDT",
    "ONTUSDT",
    "OSMOUSDT",
    "QTUMUSDT",
    "ROSEUSDT",
    "RUNEUSDT",
    "RVNUSDT",
    "SEIUSDT",
    "SUIUSDT",
    "TIAUSDT",
    "TRXUSDT",
    "VETUSDT",
    "XDCUSDT",
    "XECUSDT",
    "XEMUSDT",
    "XLMUSDT",
    "XMRUSDT",
    "XRDUSDT",
    "XRPUSDT",
    "XTZUSDT",
    "ZECUSDT",
    "ZILUSDT",
    "AKTUSDT",
    "HIVEUSDT",
    "WAXPUSDT",
]



INTERVAL = "4h"       # kline interval ("1h","4h","1d",...)
LOOKBACK_DAYS = 365   # history window
MIN_OBS = 200         # minimum overlapping observations for a valid test
PVAL_THRESHOLD = 0.05 # "significant" cointegration level

# ==========================
# DATA FETCHING
# ==========================

def binance_futures_klines(symbol, interval, start_time_ms, end_time_ms):
    """
    Fetch historical USDT-margined futures klines from Binance.
    Returns a DataFrame indexed by open_time with a 'close' column.
    """
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
        raise ValueError(f"No kline data returned for {symbol}")

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


def load_all_prices(symbols, interval, lookback_days):
    """
    Fetch close prices for all symbols in 'symbols' from Binance futures.
    Returns a dict: {symbol: Series_of_closes}
    """
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=lookback_days)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    prices = {}
    for sym in symbols:
        try:
            print(f"Fetching {sym} klines...")
            df = binance_futures_klines(sym, interval, start_ms, end_ms)
            s = df["close"].copy()
            s.name = sym
            prices[sym] = s
        except Exception as e:
            print(f"Failed to fetch {sym}: {e}")

    return prices


# ==========================
# COINTEGRATION SCAN
# ==========================

def test_pair_cointegration(series_x, series_y, min_obs=MIN_OBS):
    """
    Run Engle–Granger cointegration test between two price Series.
    Returns (pvalue, test_stat, n_obs) or (None, None, n_obs) if too short.
    """
    df_pair = pd.concat([series_x, series_y], axis=1, join="inner").dropna()
    n = len(df_pair)
    if n < min_obs:
        return None, None, n

    x = df_pair.iloc[:, 0]
    y = df_pair.iloc[:, 1]

    stat, pval, crit = coint(x, y)
    return pval, stat, n


def scan_cointegration(prices_dict):
    """
    For all symbol pairs, run cointegration test and collect results.
    Returns a DataFrame with columns:
      ['sym_x','sym_y','pvalue','test_stat','n_obs']
    """
    results = []

    symbols = sorted(prices_dict.keys())
    for sym_x, sym_y in combinations(symbols, 2):
        s_x = prices_dict[sym_x]
        s_y = prices_dict[sym_y]

        pval, stat, n = test_pair_cointegration(s_x, s_y)

        if pval is None:
            print(f"Skipping {sym_x}-{sym_y}, not enough data (n={n})")
            continue

        print(f"Tested {sym_x}-{sym_y}: p={pval:.4f}, stat={stat:.2f}, n={n}")
        results.append({
            "sym_x": sym_x,
            "sym_y": sym_y,
            "pvalue": pval,
            "test_stat": stat,
            "n_obs": n
        })

    if not results:
        return pd.DataFrame(columns=["sym_x","sym_y","pvalue","test_stat","n_obs"])

    df_res = pd.DataFrame(results)
    df_res.sort_values("pvalue", inplace=True)
    df_res.reset_index(drop=True, inplace=True)
    return df_res


# ==========================
# MAIN
# ==========================

def main():
    prices_dict = load_all_prices(L1_TOKENS, INTERVAL, LOOKBACK_DAYS)

    if not prices_dict:
        print("No price data fetched, aborting.")
        return

    print("\n=== Running pairwise cointegration tests ===")
    res = scan_cointegration(prices_dict)

    if res.empty:
        print("No valid pairs to test.")
        return

    print("\n=== All pairs sorted by p-value (lowest = strongest CI) ===")
    print(res.to_string(index=False))

    print("\n=== Pairs with p < {:.2f} (cointegration candidates) ===".format(PVAL_THRESHOLD))
    ci_candidates = res[res["pvalue"] < PVAL_THRESHOLD]
    if ci_candidates.empty:
        print("No strongly cointegrated pairs at this threshold.")
    else:
        print(ci_candidates.to_string(index=False))


if __name__ == "__main__":
    main()
