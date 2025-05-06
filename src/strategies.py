from backtesting import Strategy
from backtesting.lib import crossover


class SMASanityCheck(Strategy):
    """
    SMA-7/25/99 crossover strategy using Backtesting.py
    """

    def init(self):
        price = self.data.df["Close"]
        self.sma7 = self.I(lambda s: s.rolling(7).mean(), price)
        self.sma25 = self.I(lambda s: s.rolling(25).mean(), price)
        self.sma99 = self.I(lambda s: s.rolling(99).mean(), price)

    def next(self):
        if (
            not self.position
            and crossover(self.sma7, self.sma25)
            and self.sma7[-1] > self.sma99[-1]
        ):
            self.buy()
        elif self.position and (
            crossover(self.sma25, self.sma7) or self.sma7[-1] < self.sma99[-1]
        ):
            self.position.close()
