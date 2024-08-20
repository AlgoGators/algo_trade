import pandas as pd
import numpy as np

from typing import Optional, Callable
from functools import partial
from pnl import PnL

from .future import Future, Instrument

### Abstract Classes

class Instrument:
    def __init__(self, prices : pd.Series, name : str, multiplier : Optional[float] = None):
        self.dates = prices.index
        self.prices = prices
        self.name = name
        self.multiplier = multiplier

class Strategy:
    def __init__(self, instruments : list[Instrument]):
        self.instruments : list[Instrument] = instruments
        self.rules : list[Callable] = []
        self.scalars : list[float] = []

    @property
    def positions(self) -> pd.DataFrame:
        if not hasattr(self, '_positions'):
            self._positions = pd.DataFrame()
            for rule in self.rules:
                df = rule(self.instruments)
                self._positions = df if self._positions.empty else self._positions * df

            scalar = np.prod(self.scalars)

            self._positions = self._positions * scalar

        return self._positions
    
    @positions.setter
    def positions(self, value):
        self._positions = value

### Example Strategy

class TrendFollowing(Strategy):
    def __init__(self, instruments : list[Instrument], risk_target : float, capital : float):
        super().__init__(instruments)
        self.rules = [
            partial(risk_parity, std_fn=normal_std, risk_target=risk_target),
            partial(trend_signals, std_fn=normal_std),
            partial(equal_weight)
        ]
        self.scalars = [capital]

class TestStrategy(Strategy):
    def __init__(self, instruments : list[Instrument]):
        super().__init__(instruments)
        self.rules = [
            partial(equal_weight)
        ]
        self.scalars = [2.0]


#! Remove above

### Standard Deviation Function
def normal_std(prices : pd.Series) -> float:
    return prices.rolling(window=100).std()

### Rules

def risk_parity(instruments : list[Instrument], std_fn : Callable, risk_target  : float) -> pd.DataFrame:
    series_lst : list[pd.Series] = [(risk_target / std_fn(instrument.prices)).rename(instrument.name) for instrument in instruments]

    df = pd.DataFrame()
    for series in series_lst:
        if df.empty:
            df = series.to_frame()
        else:
            df = df.join(series.to_frame(), how='outer')

    return df

def equal_weight(instruments : list[Instrument]) -> pd.DataFrame:
    df = pd.DataFrame()
    for instrument in instruments:
        if df.empty:
            df = instrument.prices.to_frame().rename(columns={'Close': instrument.name})
        else:
            df = df.join(instrument.prices.to_frame().rename(columns={'Close': instrument.name}), how='outer')

    df.ffill(inplace=True)

    not_null_mask = df.notnull()
    weight_mask = 1 / df.notnull().sum(axis=1).astype(int)
    
    weights = not_null_mask.mul(weight_mask, axis=0)

    return weights

def trend_signals(instruments : list[Instrument], std_fn : Callable) -> pd.DataFrame:
    forecasts : list[pd.Series] = []
    for instrument in instruments:
        trend = (
            pd.DataFrame()
        )

        crossovers = [(2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256)]

        # Calculate the exponential moving averages crossovers and store them in the trend dataframe for t1, t2 in crossovers: trend[f"{t1}-{t2}"] = data["Close"].ewm(span=t1, min_periods=2).mean() - data["Close"].ewm(span=t2, min_periods=2).mean()
        for t1, t2 in crossovers:
            trend[f"{t1}-{t2}"] = (
                instrument.prices.ewm(span=t1, min_periods=2, adjust=False).mean()
                - instrument.prices.ewm(span=t2, min_periods=2, adjust=False).mean()
            )

        # Calculate the risk adjusted forecasts
        for t1, t2 in crossovers:
            trend[f"{t1}-{t2}"] /= (
                std_fn(instrument.prices) * instrument.prices)

        # Scale the crossovers by the absolute mean of all previous crossovers
        scalar_dict = {}
        for t1, t2 in crossovers:
            scalar_dict[t1] = 10 / trend[f"{t1}-{t2}"].abs().mean()

        for t1, t2 in crossovers:
            trend[f"{t1}-{t2}"] = trend[f"{t1}-{t2}"] * scalar_dict[t1]

        # Clip the scaled crossovers to -20, 20
        for t1, t2 in crossovers:
            trend[f"{t1}-{t2}"] = trend[f"{t1}-{t2}"].clip(-20, 20)

        trend["Forecast"] = 0.0

        n = len(crossovers)
        weights = {64: 1 / n, 32: 1 / n, 16: 1 / n, 8: 1 / n, 4: 1 / n, 2: 1 / n}

        for t1, t2 in crossovers:
            trend["Forecast"] += trend[f"{t1}-{t2}"] * weights[t1]

        fdm = 1.35
        trend["Forecast"] = trend["Forecast"] * fdm

        # Clip the final forecast to -20, 20
        trend["Forecast"] = trend["Forecast"].clip(-20, 20)

        forecasts.append((trend.Forecast / 10).rename(instrument.name))

    df = pd.DataFrame()
    for series in forecasts:
        if df.empty:
            df = series.to_frame()
        else:
            df = df.join(series.to_frame(), how='outer')

    return df

### Main Function (Implementation, Specific to the Example)

def main(SP500, NASDAQ):
    trend_following = TrendFollowing([SP500, NASDAQ], 0.5, 100_000)
    # print(trend_following.positions)

    # trend_carry = TrendCarry([SP500, NASDAQ], 0.5, 100_000)

def get_price_data(ticker : str, period : str) -> pd.Series:
    import yfinance
    return yfinance.download(tickers=ticker, period=period)['Close']

if __name__ == '__main__':
    SP500 = Instrument(get_price_data('^GSPC', '1y'), 'SP500')
    NASDAQ = Instrument(get_price_data('^IXIC', '5y'), 'NASDAQ')

    main(SP500, NASDAQ)