"""
Modelo LightGBM para predicción de dirección BTC.

Pipeline completo:
  1. Carga datos + features + labels
  2. Split temporal (train / validation / test)
  3. Entrena clasificador LightGBM multiclase
  4. Evalúa métricas de clasificación + métricas de trading
  5. Analiza feature importance
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix

from features import cargar_y_preparar
from labeling import triple_barrier, distribucion_labels


# =============================================================================
# Columnas que NO son features (metadata, targets, precios crudos)
# =============================================================================

COLS_EXCLUIR = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
    # Columnas del labeling
    "tp_price", "sl_price", "label", "barrier_time", "ret_operacion",
    # EMAs absolutas (usamos los ratios close/ema en su lugar)
    "ema_9", "ema_21", "ema_50", "ema_200",
    # OBV absoluto (usamos su slope)
    "obv",
    # ATR absoluto (usamos atr_pct)
    "atr_14",
    # Hora y día crudos (usamos los cíclicos sin/cos)
    "hora", "dia_semana",
]


def obtener_feature_cols(df: pd.DataFrame) -> list[str]:
    """Retorna las columnas que se usarán como features."""
    return [c for c in df.columns if c not in COLS_EXCLUIR]


# =============================================================================
# Split temporal
# =============================================================================

def split_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide el dataset en 3 periodos:
      - Train:      2017 – 2022  (aprender patrones)
      - Validation:  2022 – 2024  (ajustar hiperparámetros)
      - Test:        2024 – 2026  (evaluación final, no tocar)
    """
    df = df.copy()
    dt = pd.to_datetime(df["open_time"])

    train = df[dt < "2022-01-01"].copy()
    val = df[(dt >= "2022-01-01") & (dt < "2024-01-01")].copy()
    test = df[dt >= "2024-01-01"].copy()

    print(f"Split temporal:")
    print(f"  Train:      {len(train):>6} filas  (2017–2022)")
    print(f"  Validation: {len(val):>6} filas  (2022–2024)")
    print(f"  Test:       {len(test):>6} filas  (2024–2026)")

    return train, val, test


# =============================================================================
# Entrenamiento
# =============================================================================

def entrenar_modelo(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    params: dict | None = None,
) -> lgb.Booster:
    """
    Entrena un clasificador LightGBM multiclase (LONG / HOLD / SHORT).
    """
    # Mapear labels: -1 → 0, 0 → 1, 1 → 2
    label_map = {-1: 0, 0: 1, 1: 2}

    X_train = train[feature_cols]
    y_train = train["label"].map(label_map)
    X_val = val[feature_cols]
    y_val = val["label"].map(label_map)

    # Pesos para balancear clases desiguales
    class_counts = y_train.value_counts().sort_index()
    total = len(y_train)
    class_weights = {c: total / (len(class_counts) * count) for c, count in class_counts.items()}
    sample_weights = y_train.map(class_weights).values

    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    default_params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
    }

    if params:
        default_params.update(params)

    print(f"\nEntrenando LightGBM con {len(feature_cols)} features...")

    callbacks = [
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=50),
    ]

    model = lgb.train(
        default_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    print(f"  Mejor iteración: {model.best_iteration}")

    return model


# =============================================================================
# Evaluación
# =============================================================================

def evaluar_modelo(
    model: lgb.Booster,
    df: pd.DataFrame,
    feature_cols: list[str],
    nombre: str = "Test",
    umbral_confianza: float = 0.0,
) -> dict:
    """Evalúa el modelo y muestra métricas de clasificación + trading."""
    label_map = {-1: 0, 0: 1, 1: 2}
    inv_label_map = {0: -1, 1: 0, 2: 1}
    target_names = ["SHORT", "HOLD", "LONG"]

    X = df[feature_cols]
    y_true = df["label"].map(label_map).values

    # Predicciones
    y_proba = model.predict(X)
    y_pred = y_proba.argmax(axis=1)

    # --- Métricas de clasificación ---
    print(f"\n{'='*60}")
    print(f"EVALUACIÓN: {nombre}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    print("Matriz de confusión:")
    print(f"  {'':>10} {'SHORT':>8} {'HOLD':>8} {'LONG':>8}  ← predicho")
    for i, row_name in enumerate(target_names):
        print(f"  {row_name:>10} {cm[i][0]:>8} {cm[i][1]:>8} {cm[i][2]:>8}")

    # --- Métricas de trading ---
    df_eval = df.copy()
    df_eval["pred"] = [inv_label_map[p] for p in y_pred]
    df_eval["prob_long"] = y_proba[:, 2]

    # Solo operamos cuando el modelo predice LONG
    trades_long = df_eval[df_eval["pred"] == 1]

    # Aplicar filtro de confianza
    if umbral_confianza > 0 and len(trades_long) > 0:
        trades_long = trades_long[trades_long["prob_long"] >= umbral_confianza]

    capital_inicial = 10_000.0

    print(f"\n--- Simulación de Trading ({nombre}) ---")
    print(f"  Capital inicial: ${capital_inicial:,.2f}")
    if umbral_confianza > 0:
        print(f"  Umbral de confianza: {umbral_confianza:.0%}")

    if len(trades_long) > 0:
        wins = trades_long[trades_long["label"] == 1]
        win_rate = len(wins) / len(trades_long) * 100
        ret_medio = trades_long["ret_operacion"].mean() * 100
        ret_total = trades_long["ret_operacion"].sum() * 100
        max_dd = _max_drawdown(trades_long["ret_operacion"].values)

        # Simular equity con capital compuesto
        equity = capital_inicial * np.cumprod(1 + trades_long["ret_operacion"].values)
        capital_final = equity[-1]
        capital_minimo = equity.min()

        print(f"  Trades LONG: {len(trades_long)}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Retorno medio/trade: {ret_medio:+.3f}%")
        print(f"  Retorno acumulado: {ret_total:+.1f}%")
        print(f"  Capital final: ${capital_final:,.2f}")
        print(f"  Beneficio: ${capital_final - capital_inicial:+,.2f}")
        print(f"  Capital mínimo: ${capital_minimo:,.2f}")
        print(f"  Max Drawdown: {max_dd:.1f}%")

        rets = trades_long["ret_operacion"].values
        if rets.std() > 0:
            sharpe = (rets.mean() / rets.std()) * np.sqrt(8760)
            print(f"  Sharpe Ratio (aprox): {sharpe:.2f}")
    else:
        print("  Sin trades LONG")
        win_rate = 0
        ret_medio = 0

    print(f"{'='*60}")

    return {
        "trades_long": len(trades_long),
        "win_rate": win_rate,
        "ret_medio": ret_medio,
    }


def analizar_umbrales(
    model: lgb.Booster,
    df: pd.DataFrame,
    feature_cols: list[str],
    nombre: str = "Test",
) -> None:
    """
    Analiza el impacto de distintos umbrales de confianza.
    prob_long es la probabilidad de la clase LONG (índice 2).
    """
    X = df[feature_cols]
    y_proba = model.predict(X)

    df_eval = df.copy()
    df_eval["prob_long"] = y_proba[:, 2]
    df_eval["pred_long"] = y_proba.argmax(axis=1) == 2

    # Solo considerar filas donde el modelo predice LONG
    base_long = df_eval[df_eval["pred_long"]].copy()

    capital_inicial = 10_000.0

    print(f"\n{'='*60}")
    print(f"ANÁLISIS DE UMBRALES DE CONFIANZA — {nombre}")
    print(f"  Capital inicial: ${capital_inicial:,.2f}")
    print(f"{'='*60}")
    print(f"  {'Umbral':>8} {'Trades':>8} {'WinRate':>8} {'Ret/Trade':>10} "
          f"{'Capital$':>12} {'MaxDD':>8} {'Sharpe':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*12} {'-'*8} {'-'*8}")

    umbrales = [0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    for umbral in umbrales:
        if umbral > 0:
            trades = base_long[base_long["prob_long"] >= umbral]
        else:
            trades = base_long

        if len(trades) == 0:
            print(f"  {umbral:>7.0%} {'0':>8} {'---':>8} {'---':>10} "
                  f"{'---':>12} {'---':>8} {'---':>8}")
            continue

        wins = trades[trades["label"] == 1]
        wr = len(wins) / len(trades) * 100
        ret_m = trades["ret_operacion"].mean() * 100
        max_dd = _max_drawdown(trades["ret_operacion"].values)
        rets = trades["ret_operacion"].values
        sharpe = (rets.mean() / rets.std()) * np.sqrt(8760) if rets.std() > 0 else 0
        capital_final = capital_inicial * np.prod(1 + rets)

        print(f"  {umbral:>7.0%} {len(trades):>8} {wr:>7.1f}% {ret_m:>+9.3f}% "
              f"  ${capital_final:>10,.2f} {max_dd:>7.1f}% {sharpe:>7.2f}")

    print(f"{'='*60}")


def _max_drawdown(returns: np.ndarray) -> float:
    """Calcula el max drawdown de una serie de retornos."""
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return drawdown.min() * 100


# =============================================================================
# Feature Importance
# =============================================================================

def mostrar_importancia(model: lgb.Booster, top_n: int = 15) -> pd.DataFrame:
    """Muestra las features más importantes."""
    importance = pd.DataFrame({
        "feature": model.feature_name(),
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    print(f"\nTop {top_n} features por importancia (gain):")
    for i, row in importance.head(top_n).iterrows():
        bar = "█" * int(row["importance"] / importance["importance"].max() * 30)
        print(f"  {row['feature']:>25}: {bar} ({row['importance']:.0f})")

    return importance


# =============================================================================
# Guardar / Cargar modelo
# =============================================================================

def guardar_modelo(model: lgb.Booster, feature_cols: list[str], path: str) -> None:
    """Guarda el modelo y la lista de features."""
    model.save_model(path)
    meta_path = path.replace(".txt", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"feature_cols": feature_cols, "best_iteration": model.best_iteration}, f)
    print(f"\nModelo guardado en: {path}")


def cargar_modelo(path: str) -> tuple[lgb.Booster, list[str]]:
    """Carga el modelo y la lista de features."""
    model = lgb.Booster(model_file=path)
    meta_path = path.replace(".txt", "_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta["feature_cols"]


# =============================================================================
# Pipeline completo
# =============================================================================

def run():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "btc_1h_2016_2026.csv")

    # 1. Features
    print("1. Cargando datos y calculando features...")
    df = cargar_y_preparar(csv_path)

    # 2. Labels
    print("\n2. Aplicando Triple Barrier Method...")
    df = triple_barrier(df, tp_mult=2.0, sl_mult=1.0, max_holding=24)
    distribucion_labels(df)

    # 3. Split
    print("\n3. Split temporal...")
    train, val, test = split_temporal(df)

    # 4. Features a usar
    feature_cols = obtener_feature_cols(df)
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")

    # 5. Entrenar
    model = entrenar_modelo(train, val, feature_cols)

    # 6. Evaluar (sin filtro)
    evaluar_modelo(model, val, feature_cols, nombre="Validation (2022–2024)")
    evaluar_modelo(model, test, feature_cols, nombre="Test (2024–2026)")

    # 7. Análisis de umbrales de confianza
    analizar_umbrales(model, val, feature_cols, nombre="Validation (2022–2024)")
    analizar_umbrales(model, test, feature_cols, nombre="Test (2024–2026)")

    # 8. Feature importance
    importance = mostrar_importancia(model)

    # 9. Guardar
    model_path = os.path.join(os.path.dirname(__file__), "data", "modelo_lgbm.txt")
    guardar_modelo(model, feature_cols, model_path)

    return model, feature_cols, df


if __name__ == "__main__":
    model, feature_cols, df = run()
