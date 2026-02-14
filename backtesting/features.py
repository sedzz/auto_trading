"""
Feature Engineering Pipeline para BTC 1H.

Transforma datos OHLCV crudos en features interpretables
para modelos de ML (LightGBM).
"""

import pandas as pd
import numpy as np


# =============================================================================
# Indicadores técnicos (implementación propia para evitar dependencias extra)
# =============================================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    mid = sma(series, period)
    std = series.rolling(window=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    return (volume * direction).cumsum()


def stochastic_rsi(series: pd.Series, rsi_period: int = 14, stoch_period: int = 14) -> pd.Series:
    rsi_values = rsi(series, rsi_period)
    rsi_min = rsi_values.rolling(window=stoch_period).min()
    rsi_max = rsi_values.rolling(window=stoch_period).max()
    rsi_range = rsi_max - rsi_min
    return ((rsi_values - rsi_min) / rsi_range.replace(0, np.nan)) * 100


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr_values = atr(high, low, close, period)
    plus_di = 100 * ema(plus_dm, period) / atr_values
    minus_di = 100 * ema(minus_dm, period) / atr_values
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return ema(dx, period)


# =============================================================================
# Pipeline principal
# =============================================================================

def calcular_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe un DataFrame con columnas OHLCV y devuelve
    el mismo DataFrame con todas las features añadidas.
    """
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # --- Tendencia ---
    for p in [9, 21, 50, 200]:
        df[f"ema_{p}"] = ema(close, p)
        df[f"close_sobre_ema_{p}"] = close / df[f"ema_{p}"]

    macd_line, macd_signal, macd_hist = macd(close)
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    df["adx"] = adx(high, low, close)

    # --- Momentum ---
    df["rsi_14"] = rsi(close)
    df["stoch_rsi"] = stochastic_rsi(close)
    for p in [1, 4, 12, 24]:
        df[f"roc_{p}h"] = close.pct_change(periods=p)

    # --- Volatilidad ---
    df["atr_14"] = atr(high, low, close)
    df["atr_pct"] = df["atr_14"] / close  # ATR normalizado
    bb_upper, bb_mid, bb_lower = bollinger_bands(close)
    df["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid

    df["high_low_range"] = (high - low) / close  # rango de la vela normalizado

    # --- Volumen ---
    vol_media_20 = sma(volume, 20)
    df["vol_relativo"] = volume / vol_media_20.replace(0, np.nan)
    df["obv"] = obv(close, volume)
    df["obv_slope"] = df["obv"].pct_change(periods=12)  # pendiente OBV 12h

    # --- Returns pasados ---
    for p in [1, 4, 12, 24, 168]:  # 168h = 7 días
        df[f"ret_{p}h"] = close.pct_change(periods=p)

    # --- Temporales ---
    if "open_time" in df.columns:
        dt = pd.to_datetime(df["open_time"])
    elif df.index.dtype == "datetime64[ns]" or hasattr(df.index, "hour"):
        dt = df.index.to_series()
    else:
        dt = None

    if dt is not None:
        df["hora"] = dt.dt.hour
        df["dia_semana"] = dt.dt.dayofweek
        # Encoding cíclico para que el modelo entienda que hora 23 y hora 0 son cercanas
        df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
        df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
        df["dia_sin"] = np.sin(2 * np.pi * df["dia_semana"] / 7)
        df["dia_cos"] = np.cos(2 * np.pi * df["dia_semana"] / 7)

    return df


def cargar_y_preparar(csv_path: str, dropna: bool = True) -> pd.DataFrame:
    """
    Carga el CSV, calcula features y limpia NaNs iniciales
    (producidos por ventanas de indicadores).
    """
    df = pd.read_csv(csv_path, parse_dates=["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)

    # Convertir columnas numéricas
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = calcular_features(df)

    if dropna:
        df = df.dropna().reset_index(drop=True)

    return df


# =============================================================================
# Ejecución directa para verificar
# =============================================================================

if __name__ == "__main__":
    import os

    csv_path = os.path.join(os.path.dirname(__file__), "data", "btc_1h_2016_2026.csv")
    df = cargar_y_preparar(csv_path)

    print(f"Filas originales: 74001 → Filas con features: {len(df)}")
    print(f"Columnas totales: {len(df.columns)}")
    print(f"\nFeatures generadas:")
    cols_nuevas = [c for c in df.columns if c not in [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]]
    for c in cols_nuevas:
        print(f"  {c}: min={df[c].min():.4f}, max={df[c].max():.4f}, nulls={df[c].isna().sum()}")

    print(f"\nPrimeras 3 filas (features seleccionadas):")
    print(df[["open_time", "close", "rsi_14", "macd", "atr_pct", "bb_position", "vol_relativo", "ret_24h"]].head(3).to_string())
