import requests
import pandas as pd
import time

symbol = "BTCUSDT"
interval = "1h"
start_ts = int(pd.Timestamp("2016-01-01").timestamp() * 1000)
end_ts = int(pd.Timestamp("2026-01-01").timestamp() * 1000)

url = "https://api.binance.com/api/v3/klines"
data = []

while start_ts < end_ts:
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ts,
        "limit": 1000
    }
    r = requests.get(url, params=params).json()
    if not r:
        break

    data.extend(r)
    start_ts = r[-1][0] + 1
    time.sleep(0.5)

cols = [
    "open_time","open","high","low","close","volume",
    "close_time","quote_asset_volume","num_trades",
    "taker_buy_base","taker_buy_quote","ignore"
]

df = pd.DataFrame(data, columns=cols)
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
df.to_csv("btc_1h_2016_2026.csv", index=False)
