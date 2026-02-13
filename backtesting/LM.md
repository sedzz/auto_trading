1. Agente ML (Análisis Técnico)

  Tu data de 7.5 años con OHLCV es ideal para estos modelos:
  Algoritmo: XGBoost / LightGBM
  Para qué: Dirección (clasificación) + TP/SL (regresión)
  Por qué: El estándar en trading cuantitativo. Maneja bien features tabulares, rápido, interpretable
  ────────────────────────────────────────
  Algoritmo: Random Forest
  Para qué: Baseline sólido
  Por qué: Menos overfitting que un solo árbol, buen punto de partida
  ────────────────────────────────────────
  Algoritmo: LSTM / GRU
  Para qué: Capturar patrones temporales
  Por qué: Útil si quieres usar secuencias de precios (ventanas de N velas)
  ────────────────────────────────────────
  Algoritmo: Transformer temporal (TFT)
  Para qué: Series temporales avanzadas
  Por qué: Estado del arte pero necesita más datos y tuning
  Recomendación: Empieza con LightGBM para clasificación (buy/sell/hold) y regresión (TP/SL). Es el que mejor
  balance tiene entre rendimiento y complejidad.

  Features típicos a calcular:
  - RSI, MACD, Bollinger Bands, ATR
  - EMAs (9, 21, 50, 200)
  - Volumen relativo, OBV
  - Patrones de velas (Doji, Hammer, etc.)
  - Retornos en ventanas (1d, 3d, 7d, 14d, 30d)