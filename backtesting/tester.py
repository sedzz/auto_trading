from backtesting import Backtest, Strategy

class MLStrategy(Strategy):
    def init(self):
        # Tu modelo ML aquí
        pass
    
    def next(self):
        # Lógica de decisión
        prediction = ml_model.predict(...)
        if prediction > 0.7:
            self.buy(sl=0.98, tp=1.03)

bt = Backtest(data, MLStrategy, cash=100000, commission=.001)
stats = bt.run()
bt.plot()