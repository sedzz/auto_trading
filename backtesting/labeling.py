"""
Triple Barrier Labeling para BTC 1H.

Asigna a cada vela un label basado en qué barrera se toca primero:
  1 (LONG)  → el precio subió hasta el Take Profit antes que el Stop Loss
  -1 (SHORT) → el precio bajó hasta el Stop Loss antes que el Take Profit
  0 (HOLD)  → ninguna barrera se tocó dentro del tiempo máximo

Esto es más realista que un simple "subió/bajó" porque simula
cómo opera un trader real: con TP, SL y tiempo límite.
"""

import pandas as pd
import numpy as np


def triple_barrier(
    df: pd.DataFrame,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_holding: int = 24,
    atr_col: str = "atr_14",
) -> pd.DataFrame:
    """
    Aplica el Triple Barrier Method a cada fila del DataFrame.

    Parámetros
    ----------
    df : DataFrame
        Debe contener columnas 'close' y `atr_col`.
    tp_mult : float
        Multiplicador de ATR para el Take Profit. Default 2.0 (2x ATR).
    sl_mult : float
        Multiplicador de ATR para el Stop Loss. Default 1.0 (1x ATR).
    max_holding : int
        Máximo de velas (horas) antes de cerrar la operación. Default 24h.
    atr_col : str
        Nombre de la columna ATR a usar para las barreras dinámicas.

    Retorna
    -------
    DataFrame con columnas añadidas:
        - tp_price: precio del Take Profit
        - sl_price: precio del Stop Loss
        - label: 1 (long), -1 (short), 0 (hold/timeout)
        - barrier_time: en cuántas velas se resolvió
        - ret_operacion: retorno de la operación resuelta
    """
    df = df.copy()
    closes = df["close"].values
    atrs = df[atr_col].values
    n = len(df)

    labels = np.zeros(n, dtype=np.int8)
    barrier_times = np.zeros(n, dtype=np.int16)
    ret_operacion = np.zeros(n, dtype=np.float64)
    tp_prices = np.zeros(n, dtype=np.float64)
    sl_prices = np.zeros(n, dtype=np.float64)

    for i in range(n):
        entry = closes[i]
        atr_val = atrs[i]

        tp_price = entry + atr_val * tp_mult
        sl_price = entry - atr_val * sl_mult
        tp_prices[i] = tp_price
        sl_prices[i] = sl_price

        # Ventana de futuro disponible
        end = min(i + max_holding + 1, n)
        future_closes = closes[i + 1 : end]

        if len(future_closes) == 0:
            labels[i] = 0
            barrier_times[i] = 0
            ret_operacion[i] = 0.0
            continue

        resolved = False
        for j, price in enumerate(future_closes, start=1):
            if price >= tp_price:
                labels[i] = 1   # LONG ganador
                barrier_times[i] = j
                ret_operacion[i] = (tp_price - entry) / entry
                resolved = True
                break
            elif price <= sl_price:
                labels[i] = -1  # SHORT / SL tocado
                barrier_times[i] = j
                ret_operacion[i] = (sl_price - entry) / entry
                resolved = True
                break

        if not resolved:
            # Timeout: se cierra al final de max_holding
            exit_price = future_closes[-1]
            ret = (exit_price - entry) / entry
            # Si el retorno es positivo asignamos 1, negativo -1, ~0 hold
            if ret > 0.002:
                labels[i] = 1
            elif ret < -0.002:
                labels[i] = -1
            else:
                labels[i] = 0
            barrier_times[i] = len(future_closes)
            ret_operacion[i] = ret

    df["tp_price"] = tp_prices
    df["sl_price"] = sl_prices
    df["label"] = labels
    df["barrier_time"] = barrier_times
    df["ret_operacion"] = ret_operacion

    return df


def distribucion_labels(df: pd.DataFrame) -> None:
    """Imprime la distribución de labels y estadísticas clave."""
    total = len(df)
    print("=" * 50)
    print("DISTRIBUCIÓN DE LABELS")
    print("=" * 50)

    for label, nombre in [(1, "LONG"), (0, "HOLD"), (-1, "SHORT/SL")]:
        subset = df[df["label"] == label]
        count = len(subset)
        pct = count / total * 100
        ret_medio = subset["ret_operacion"].mean() * 100
        tiempo_medio = subset["barrier_time"].mean()
        print(f"  {nombre:>10}: {count:>6} ({pct:5.1f}%)  "
              f"ret_medio={ret_medio:+.2f}%  tiempo_medio={tiempo_medio:.1f}h")

    print(f"\n  Total: {total}")
    print(f"  Ratio Long/Short: {len(df[df['label']==1])}/{len(df[df['label']==-1])}")

    # Expectativa matemática
    ret_medio_total = df["ret_operacion"].mean() * 100
    ret_std = df["ret_operacion"].std() * 100
    print(f"\n  Retorno medio por operación: {ret_medio_total:+.3f}%")
    print(f"  Desviación estándar: {ret_std:.3f}%")
    print("=" * 50)


# =============================================================================
# Ejecución directa para verificar
# =============================================================================

if __name__ == "__main__":
    import os
    from features import cargar_y_preparar

    csv_path = os.path.join(os.path.dirname(__file__), "data", "btc_1h_2016_2026.csv")
    df = cargar_y_preparar(csv_path)

    print("Aplicando Triple Barrier Method...")
    print(f"  TP = 2x ATR, SL = 1x ATR, Max holding = 24h\n")

    df = triple_barrier(df, tp_mult=2.0, sl_mult=1.0, max_holding=24)

    distribucion_labels(df)

    print(f"\nEjemplo de 5 filas:")
    print(df[["open_time", "close", "atr_14", "tp_price", "sl_price",
              "label", "barrier_time", "ret_operacion"]].head(5).to_string())
