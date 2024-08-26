import pandas as pd
import numpy as np
from abc import ABC

from typing import Callable
from functools import partial

from algo_trade.instrument import Future, Instrument, RollType, ContractType, Agg
from algo_trade.risk_measures import RiskMeasure, GARCH

DAYS_IN_YEAR = 256

class Strategy(ABC):
    """
    Strategy class is an abstract class that defines the structure of a strategy. Strategies are composed of a list of Insturments, a list of Callable rules, and a list of scalars. The rules are applied to the instruments to generate a DataFrame of positions. The scalars are applied to the positions to generate the final positions DataFrame.
    """

    def __init__(self, risk_target: float, capital: float):
        self.instruments: list[Instrument] = []
        self._risk_target = risk_target
        self._capital = capital
        self.risk_object : RiskMeasure = RiskMeasure()
        self.rules: list[Callable] = []
        self.scalars: list[float] = []

    @property
    def positions(self) -> pd.DataFrame:
        if not hasattr(self, "_positions"):
            self._positions = pd.DataFrame()
            for rule in self.rules:
                df : pd.DataFrame = rule()
                self._positions = df if self._positions.empty else self._positions * df

            scalar = np.prod(self.scalars)

            self._positions = self._positions * scalar

        return self._positions

    @positions.setter
    def positions(self, value):
        self._positions = value

    def fetch_data(self):
        """
        The Fetch data method is the a required initialization step within designing a strategy. This method is used to fetch the data for the instruments within the strategy. It is strategy specific and should be implemented by the user.
        """
        raise NotImplementedError("Fetch Data method not implemented")


### Example Strategy


class TrendFollowing(Strategy):
    def __init__(self, instruments: list[Future], risk_target: float, capital: float):
        super().__init__(risk_target=risk_target, capital=capital)
        # Overload the instruments
        self.instruments: list[Future] = instruments
        self.risk_object = GARCH(
            instruments=instruments,
            weights=(0.01, 0.01, 0.98),
            minimum_observations=100
        )
        self.rules = [
            partial(risk_parity, risk_object=self.risk_object, risk_target=risk_target),
            partial(trend_signals, instruments=instruments, risk_object=self.risk_object),
            partial(equal_weight, instruments=instruments),
            partial(capital_scaling, instruments=instruments, capital=capital)
        ]
        self.scalars = []
        self.fetch_data()  # Fetch the data for the instruments

    def fetch_data(self) -> None:
        """
        The Fetch data method for the Trend Following strategy is requires the following instrument specific data:
        1. Prices(Open, High, Low, Close, Volume)
        2. Backadjusted Prices (Close)
        """
        # Load the front calendar contract data with a daily aggregation
        [instrument.add_data(Agg.DAILY, RollType.CALENDAR, ContractType.FRONT) for instrument in self.instruments]

### Rules

def capital_scaling(instruments: list[Future], capital: float) -> pd.DataFrame:
    df = pd.DataFrame()
    instrument: Future
    for instrument in instruments:
        if df.empty:
            df = (
                instrument.front
                .get_close()
                .to_frame(instrument.name)
            )
        else:
            df = df.join(
                instrument.front
                .get_close()
                .to_frame(instrument.name),
                how="outer",
            )

    df.ffill(inplace=True)

    capital_weighting = capital / df

    return capital_weighting

def risk_parity(risk_object: RiskMeasure, risk_target: float) -> pd.DataFrame:
    return risk_target / risk_object.get_var().to_standard_deviation().annualize()

def equal_weight(instruments: list[Future]) -> pd.DataFrame:
    df = pd.DataFrame()
    instrument: Future
    for instrument in instruments:
        if df.empty:
            df = (
                instrument.front
                .get_close()
                .to_frame(instrument.name)
            )
        else:
            df = df.join(
                instrument.front
                .get_close()
                .to_frame(instrument.name),
                how="outer",
            )

    df.ffill(inplace=True)

    not_null_mask = df.notnull()
    weight_mask = 1 / df.notnull().sum(axis=1).astype(int)

    weights = not_null_mask.mul(weight_mask, axis=0)

    return weights


def trend_signals(instruments: list[Future], risk_object : RiskMeasure) -> pd.DataFrame:
    forecasts: list[pd.Series] = []
    instrument: Future
    for instrument in instruments:
        trend = pd.DataFrame()

        crossovers = [(2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256)]

        # Calculate the exponential moving averages crossovers and store them in the trend dataframe for t1, t2 in crossovers: trend[f"{t1}-{t2}"] = data["Close"].ewm(span=t1, min_periods=2).mean() - data["Close"].ewm(span=t2, min_periods=2).mean()
        for t1, t2 in crossovers:
            trend[f"{t1}-{t2}"] = (
                instrument.front
                .get_backadjusted()
                .ewm(span=t1, min_periods=2, adjust=False)
                .mean()
                - instrument.front
                .get_backadjusted()
                .ewm(span=t2, min_periods=2, adjust=False)
                .mean()
            )

        # Calculate the risk adjusted forecasts
        for t1, t2 in crossovers:
            trend[f"{t1}-{t2}"] /= (
                risk_object
                .get_var()
                .to_standard_deviation()
                .annualize()
                [instrument.get_symbol()]
                * instrument.front.close
            )

        # Scale the crossovers by the absolute mean of all previous crossovers
        scalar_dict = {}
        for t1, t2 in crossovers:
            scalar_dict[t1] = 10 / trend[f"{t1}-{t2}"].abs().mean()

        for t1, t2 in crossovers:
            trend[f"{t1}-{t2}"] = trend[f"{t1}-{t2}"] * scalar_dict[t1]

        # Clip the scaled crossovers to -20, 20
        for t1, t2 in crossovers:
            trend[f"{t1}-{t2}"] = trend[f"{t1}-{t2}"].clip(-20, 20)

        trend.Forecast = 0.0

        n = len(crossovers)
        weights = {64: 1 / n, 32: 1 / n, 16: 1 / n, 8: 1 / n, 4: 1 / n, 2: 1 / n}

        for t1, t2 in crossovers:
            trend.Forecast += trend[f"{t1}-{t2}"] * weights[t1]

        fdm = 1.35
        trend.Forecast = trend.Forecast * fdm

        # Clip the final forecast to -20, 20
        trend.Forecast = trend.Forecast.clip(-20, 20)

        forecasts.append((trend.Forecast / 10).rename(instrument.name))

    df = pd.DataFrame()
    for series in forecasts:
        if df.empty:
            df = series.to_frame()
        else:
            df = df.join(series.to_frame(), how="outer")

    return df


### Main Function (Implementation, Specific to the Example)


def main():
    instruments: list[Future] = [
        Future(symbol="ES", dataset="CME", multiplier=5)
    ]
    trend_following: TrendFollowing = TrendFollowing(instruments, 0.2, 100_000)
    positions: pd.DataFrame = trend_following.positions
    print(positions)


if __name__ == "__main__":
    main()
