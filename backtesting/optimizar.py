"""
Optimización de hiperparámetros con Optuna.

Busca la mejor combinación de hiperparámetros de LightGBM
maximizando el Sharpe ratio en validation con umbral 75%.
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

from features import cargar_y_preparar
from labeling import triple_barrier, distribucion_labels
from model import (
    obtener_feature_cols, split_temporal, entrenar_modelo,
    analizar_umbrales, mostrar_importancia, guardar_modelo,
    _max_drawdown,
)


def objetivo(trial, train, val, feature_cols):
    """Función objetivo para Optuna: maximizar Sharpe en validation."""

    params = {
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
    }

    # Entrenar modelo (silencioso)
    label_map = {-1: 0, 0: 1, 1: 2}
    X_train = train[feature_cols]
    y_train = train["label"].map(label_map)
    X_val = val[feature_cols]
    y_val = val["label"].map(label_map)

    class_counts = y_train.value_counts().sort_index()
    total = len(y_train)
    class_weights = {c: total / (len(class_counts) * count) for c, count in class_counts.items()}
    sample_weights = y_train.map(class_weights).values

    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    full_params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "verbose": -1,
        **params,
    }

    model = lgb.train(
        full_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),  # silencioso
        ],
    )

    # Evaluar con umbral 75% en validation
    y_proba = model.predict(X_val)
    df_eval = val.copy()
    df_eval["prob_long"] = y_proba[:, 2]
    df_eval["pred_long"] = y_proba.argmax(axis=1) == 2

    trades = df_eval[df_eval["pred_long"] & (df_eval["prob_long"] >= 0.75)]

    if len(trades) < 30:
        # Muy pocos trades, no fiable
        return -999

    wins = trades[trades["label"] == 1]
    wr = len(wins) / len(trades)
    rets = trades["ret_operacion"].values

    if rets.std() == 0:
        return -999

    sharpe = (rets.mean() / rets.std()) * np.sqrt(8760)
    max_dd = _max_drawdown(rets)

    # Penalizar si el drawdown es mayor al 25%
    if max_dd < -25:
        sharpe *= 0.5

    return sharpe


def run():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "btc_1h_2016_2026.csv")

    # 1. Preparar datos
    print("1. Cargando datos y calculando features...")
    df = cargar_y_preparar(csv_path)
    df = triple_barrier(df, tp_mult=2.0, sl_mult=1.0, max_holding=24)

    print("\n2. Split temporal...")
    train, test = split_temporal(df)
    feature_cols = obtener_feature_cols(df)

    # Validation interna: último 10% del train para Optuna
    split_idx = int(len(train) * 0.9)
    train_inner = train.iloc[:split_idx].copy()
    val = train.iloc[split_idx:].copy()

    # 2. Optimizar
    print(f"\n3. Optimizando con Optuna (100 trials)...")
    print(f"   Objetivo: maximizar Sharpe ratio en validation (umbral 75%)\n")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objetivo(trial, train_inner, val, feature_cols),
        n_trials=100,
        show_progress_bar=True,
    )

    # 3. Resultados
    print(f"\n{'='*60}")
    print(f"RESULTADOS DE OPTIMIZACIÓN")
    print(f"{'='*60}")
    print(f"  Mejor Sharpe (val, 75%): {study.best_value:.2f}")
    print(f"  Mejores hiperparámetros:")
    for k, v in study.best_params.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    # 4. Entrenar modelo final con todo el train + mejores params
    print(f"\n4. Entrenando modelo final con mejores hiperparámetros...")
    best_model = entrenar_modelo(train, feature_cols, params=study.best_params)

    # 5. Comparar con baseline
    print(f"\n5. Comparación BASELINE vs OPTIMIZADO:")
    print(f"\n--- MODELO OPTIMIZADO ---")
    analizar_umbrales(best_model, train, feature_cols, nombre="Train (2017–2025)")
    analizar_umbrales(best_model, test, feature_cols, nombre="Test (2025–2026)")

    # 6. Feature importance
    mostrar_importancia(best_model)

    # 7. Guardar
    model_path = os.path.join(os.path.dirname(__file__), "data", "modelo_lgbm_optuna.txt")
    guardar_modelo(best_model, feature_cols, model_path)

    # 8. Top 5 trials
    print(f"\nTop 5 trials:")
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value else -999, reverse=True)
    for i, t in enumerate(trials_sorted[:5]):
        print(f"  #{t.number}: Sharpe={t.value:.2f} | "
              f"leaves={t.params['num_leaves']}, lr={t.params['learning_rate']:.3f}, "
              f"depth={t.params['max_depth']}")

    return best_model, study


if __name__ == "__main__":
    best_model, study = run()
