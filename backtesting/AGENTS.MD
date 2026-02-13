# 🤖 Backtesting de Algoritmos ML

## 🎯 Objetivo

Desarrollar un algoritmo de Machine Learning para trading de Bitcoin que:

### Entrada de Datos
- **Precio actual**: Datos en tiempo real del mercado
- **Histórico**: Últimos **10 años** de datos (OHLCV)
- **Features técnicos**: Indicadores calculados sobre la ventana histórica

### Salidas del Modelo
1. **Dirección predicha**: Alcista / Bajista / Neutral (probabilidad)
2. **Take Profit (TP)**: Nivel de precio para cerrar con ganancia
3. **Stop Loss (SL)**: Nivel de precio para limitar pérdidas

### Métricas de Éxito
- Win rate > 55%
- Risk/Reward ratio mínimo 1:2
- Drawdown máximo < 20%
- Sharpe ratio > 1.0

## 📁 Estructura

```
backtesting/
├── tester.py          # Script base con backtesting.py
├── data/
│   └── btc_10y.csv    # Datos históricos 10 años
├── models/
│   └── predictor.pkl  # Modelo entrenado
└── results/
    └── backtests/     # Reportes de pruebas
```

## 🔄 Flujo del Algoritmo

```
[Datos 10 años] → [Feature Engineering] → [ML Model] → [Predicción]
                                              ↓
                                    [Dirección, TP, SL] → [Backtest]
```

## 📝 TODO

- [ ] Recopilar datos históricos BTC (10 años)
- [ ] Definir features técnicos (RSI, MACD, Bollinger, etc.)
- [ ] Entrenar modelo de clasificación (dirección)
- [ ] Entrenar modelo de regresión (niveles TP/SL)
- [ ] Integrar con `tester.py`
- [ ] Optimizar hiperparámetros
- [ ] Validar en out-of-sample
