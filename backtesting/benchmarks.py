"""
Benchmarks para validar que el modelo aporta valor real.

Compara el modelo contra:
  1. Buy & Hold — comprar y mantener BTC
  2. Random — elegir velas al azar con mismo TP/SL
  3. All Trades — operar TODAS las velas con mismo TP/SL (sin modelo)
  4. Modelo LONG — solo cuando predice LONG
  5. Modelo SHORT — solo cuando predice SHORT

Si el modelo no supera al random, no tiene edge real.
"""

import os
import numpy as np
import pandas as pd

from features import cargar_y_preparar
from labeling import triple_barrier, distribucion_labels
from model import (
    obtener_feature_cols, split_temporal, entrenar_modelo,
    _max_drawdown,
)


def benchmark_buy_and_hold(df: pd.DataFrame, nombre: str) -> dict:
    """Simplemente comprar BTC al inicio y vender al final."""
    precio_inicio = df["close"].iloc[0]
    precio_fin = df["close"].iloc[-1]
    ret = (precio_fin - precio_inicio) / precio_inicio

    # Max drawdown del precio
    precios = df["close"].values
    peak = np.maximum.accumulate(precios)
    dd = (precios - peak) / peak
    max_dd = dd.min() * 100

    capital_inicial = 10_000.0
    capital_final = capital_inicial * (1 + ret)

    print(f"  Buy & Hold ({nombre}):")
    print(f"    Precio inicio: ${precio_inicio:,.2f}")
    print(f"    Precio fin:    ${precio_fin:,.2f}")
    print(f"    Retorno:       {ret*100:+.1f}%")
    print(f"    Capital:       ${capital_inicial:,.2f} → ${capital_final:,.2f}")
    print(f"    Max Drawdown:  {max_dd:.1f}%")

    return {"ret": ret, "capital": capital_final, "max_dd": max_dd}


def benchmark_all_trades(df: pd.DataFrame, nombre: str) -> dict:
    """Operar TODAS las velas con el mismo TP/SL, sin modelo."""
    rets = df["ret_operacion"].values
    wins = (df["label"] == 1).sum()
    total = len(df)
    wr = wins / total * 100

    capital_inicial = 10_000.0
    equity = capital_inicial * np.cumprod(1 + rets)
    capital_final = equity[-1]
    max_dd = _max_drawdown(rets)

    # Sharpe
    sharpe = (rets.mean() / rets.std()) * np.sqrt(8760) if rets.std() > 0 else 0

    print(f"  All Trades ({nombre}):")
    print(f"    Trades:        {total}")
    print(f"    Win Rate:      {wr:.1f}%")
    print(f"    Ret medio:     {rets.mean()*100:+.3f}%")
    print(f"    Capital:       ${capital_inicial:,.2f} → ${capital_final:,.2f}")
    print(f"    Max Drawdown:  {max_dd:.1f}%")
    print(f"    Sharpe:        {sharpe:.2f}")

    return {"trades": total, "wr": wr, "capital": capital_final, "max_dd": max_dd, "sharpe": sharpe}


def benchmark_random(df: pd.DataFrame, nombre: str, n_trades: int = 301, n_sims: int = 1000) -> dict:
    """
    Elegir N velas al azar y operar con el mismo TP/SL.
    Repite 1000 veces para obtener distribución estadística.
    """
    rets_all = df["ret_operacion"].values
    labels_all = df["label"].values
    capital_inicial = 10_000.0

    resultados = []
    for _ in range(n_sims):
        idx = np.random.choice(len(df), size=min(n_trades, len(df)), replace=False)
        idx.sort()  # mantener orden temporal
        rets = rets_all[idx]
        labels = labels_all[idx]

        wins = (labels == 1).sum()
        wr = wins / len(labels) * 100
        equity = capital_inicial * np.cumprod(1 + rets)
        capital_final = equity[-1]
        max_dd = _max_drawdown(rets)
        sharpe = (rets.mean() / rets.std()) * np.sqrt(8760) if rets.std() > 0 else 0

        resultados.append({
            "wr": wr, "capital": capital_final,
            "max_dd": max_dd, "sharpe": sharpe,
        })

    res = pd.DataFrame(resultados)

    print(f"  Random {n_trades} trades × {n_sims} sims ({nombre}):")
    print(f"    Win Rate:      {res['wr'].mean():.1f}% (±{res['wr'].std():.1f}%)")
    print(f"    Capital medio: ${res['capital'].mean():,.2f}")
    print(f"    Capital p5-p95: ${res['capital'].quantile(0.05):,.2f} — ${res['capital'].quantile(0.95):,.2f}")
    print(f"    Max DD medio:  {res['max_dd'].mean():.1f}%")
    print(f"    Sharpe medio:  {res['sharpe'].mean():.2f} (±{res['sharpe'].std():.2f})")

    return {
        "wr_mean": res["wr"].mean(),
        "capital_mean": res["capital"].mean(),
        "capital_p5": res["capital"].quantile(0.05),
        "capital_p95": res["capital"].quantile(0.95),
        "max_dd_mean": res["max_dd"].mean(),
        "sharpe_mean": res["sharpe"].mean(),
    }


def benchmark_modelo(model, df, feature_cols, umbral, nombre, direccion="LONG"):
    """Evalúa el modelo en una dirección (LONG o SHORT) con umbral."""
    label_map = {-1: 0, 0: 1, 1: 2}
    inv_label_map = {0: -1, 1: 0, 2: 1}

    X = df[feature_cols]
    y_proba = model.predict(X)

    df_eval = df.copy()
    df_eval["pred"] = [inv_label_map[p] for p in y_proba.argmax(axis=1)]
    df_eval["prob_long"] = y_proba[:, 2]
    df_eval["prob_short"] = y_proba[:, 0]

    capital_inicial = 10_000.0

    if direccion == "LONG":
        trades = df_eval[df_eval["pred"] == 1]
        if umbral > 0:
            trades = trades[trades["prob_long"] >= umbral]
        wins = trades[trades["label"] == 1]
    else:
        trades = df_eval[df_eval["pred"] == -1]
        if umbral > 0:
            trades = trades[trades["prob_short"] >= umbral]
        # Para SHORT, invertimos los retornos (ganamos cuando baja)
        trades = trades.copy()
        trades["ret_operacion"] = -trades["ret_operacion"]
        wins = trades[trades["ret_operacion"] > 0]

    if len(trades) == 0:
        print(f"  Modelo {direccion} umbral {umbral:.0%} ({nombre}): Sin trades")
        return None

    wr = len(wins) / len(trades) * 100
    rets = trades["ret_operacion"].values
    equity = capital_inicial * np.cumprod(1 + rets)
    capital_final = equity[-1]
    max_dd = _max_drawdown(rets)
    sharpe = (rets.mean() / rets.std()) * np.sqrt(8760) if rets.std() > 0 else 0

    print(f"  Modelo {direccion} umbral {umbral:.0%} ({nombre}):")
    print(f"    Trades:        {len(trades)}")
    print(f"    Win Rate:      {wr:.1f}%")
    print(f"    Ret medio:     {rets.mean()*100:+.3f}%")
    print(f"    Capital:       ${capital_inicial:,.2f} → ${capital_final:,.2f}")
    print(f"    Max Drawdown:  {max_dd:.1f}%")
    print(f"    Sharpe:        {sharpe:.2f}")

    return {"trades": len(trades), "wr": wr, "capital": capital_final, "max_dd": max_dd, "sharpe": sharpe}


def run():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "btc_1h_2016_2026.csv")

    # Preparar
    print("Preparando datos...\n")
    df = cargar_y_preparar(csv_path)
    df = triple_barrier(df, tp_mult=2.0, sl_mult=1.0, max_holding=24)
    train, test = split_temporal(df)
    feature_cols = obtener_feature_cols(df)

    # Entrenar modelo
    print()
    model = entrenar_modelo(train, feature_cols)

    # =============================================
    # BENCHMARKS
    # =============================================
    for nombre, subset in [("Train 2017-2025", train), ("Test 2025-2026", test)]:
        print(f"\n{'='*60}")
        print(f"BENCHMARKS — {nombre}")
        print(f"{'='*60}\n")

        # 1. Buy & Hold
        bh = benchmark_buy_and_hold(subset, nombre)

        print()

        # 2. All trades (operar todo sin modelo)
        at = benchmark_all_trades(subset, nombre)

        print()

        # 3. Random (misma cantidad de trades que el modelo con umbral 75%)
        # Primero calculamos cuántos trades hace el modelo
        X = subset[feature_cols]
        y_proba = model.predict(X)
        n_trades_modelo = ((y_proba.argmax(axis=1) == 2) & (y_proba[:, 2] >= 0.75)).sum()
        rand = benchmark_random(subset, nombre, n_trades=max(n_trades_modelo, 30))

        print()

        # 4. Modelo LONG (umbral 75%)
        ml = benchmark_modelo(model, subset, feature_cols, 0.75, nombre, "LONG")

        print()

        # 5. Modelo SHORT (umbral 75%)
        ms = benchmark_modelo(model, subset, feature_cols, 0.75, nombre, "SHORT")

        # =============================================
        # RESUMEN COMPARATIVO
        # =============================================
        print(f"\n  {'─'*55}")
        print(f"  RESUMEN — {nombre}")
        print(f"  {'─'*55}")
        print(f"  {'Estrategia':<25} {'Capital':>12} {'MaxDD':>8} {'Sharpe':>8}")
        print(f"  {'─'*25} {'─'*12} {'─'*8} {'─'*8}")

        print(f"  {'Buy & Hold':<25} ${bh['capital']:>10,.2f} {bh['max_dd']:>7.1f}%{'---':>8}")
        print(f"  {'All Trades (sin modelo)':<25} ${at['capital']:>10,.2f} {at['max_dd']:>7.1f}% {at['sharpe']:>7.2f}")
        print(f"  {'Random (media 1000 sims)':<25} ${rand['capital_mean']:>10,.2f} {rand['max_dd_mean']:>7.1f}% {rand['sharpe_mean']:>7.2f}")

        if ml:
            print(f"  {'Modelo LONG 75%':<25} ${ml['capital']:>10,.2f} {ml['max_dd']:>7.1f}% {ml['sharpe']:>7.2f}")
        if ms:
            print(f"  {'Modelo SHORT 75%':<25} ${ms['capital']:>10,.2f} {ms['max_dd']:>7.1f}% {ms['sharpe']:>7.2f}")

        print(f"  {'─'*55}")

        # Veredicto
        if ml:
            supera_random = ml["capital"] > rand["capital_p95"]
            supera_bh = ml["capital"] > bh["capital"]
            print(f"\n  ¿Modelo LONG supera random p95? {'SÍ' if supera_random else 'NO'}")
            print(f"  ¿Modelo LONG supera Buy & Hold? {'SÍ' if supera_bh else 'NO'}")


if __name__ == "__main__":
    run()
