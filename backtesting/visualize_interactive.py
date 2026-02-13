#!/usr/bin/env python3
"""
Visualización INTERACTIVA del histórico BTC usando backtesting.py
"""

import pandas as pd
import webbrowser
import os
from backtesting import Backtest, Strategy
from backtesting.lib import FractionalBacktest
import datetime

# Cargar datos
csv_path = os.path.join(os.path.dirname(__file__), 'data', 'btc_processed.csv')
df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)

# ⚠️ FIX: Ordenar de más antiguo a más reciente (backtesting.py lo requiere)
df = df.sort_index(ascending=True)

print("📊 Generando gráfico INTERACTIVO...")
print(f"Datos: {len(df)} días ({df.index.min().date()} a {df.index.max().date()})")

# Estrategia simple
class BuyAndHold(Strategy):
    def init(self):
        pass
    
    def next(self):
        if len(self.trades) == 0:
            self.buy(size=0.1)  # Comprar 0.1 BTC en lugar de 1 entero

# ⚠️ FIX: Usar más cash y cerrar trades al final
bt = Backtest(
    df, 
    BuyAndHold, 
    cash=1_000_000,  # $1M para cubrir precios altos de BTC
    commission=0.001,
    finalize_trades=True  # Cierra trades abiertos al final
)

stats = bt.run()

# Guardar y abrir
output_path = os.path.abspath(f'backtesting/exports/btc_interactive_chart_{datetime.datetime.now()}.html')
bt.plot(filename=output_path, open_browser=False)

print(f"✅ Gráfico guardado: {output_path}")
webbrowser.open(f'file://{output_path}')
