import pandas as pd
import numpy as np

from typing import Callable
from functools import partial

from future import Future, Instrument, RollType, ContractType, Agg

### Abstract Classes

# class Instrument:
#     def __init__(self, prices : pd.Series, name : str, multiplier : Optional[float] = None):
#         self.dates = prices.index
#         self.prices = prices
#         self.name = name
#         self.multiplier = multiplier
#
#     # Vol is Work in Progress
#     @property
#     def volatility(self) -> float:
#         if not hasattr(self, '_volatility'):
#             return None
#         return self._volatility
#
#     def set_volatility(self, std_fn : Callable):
#         self._volatility = std_fn(self.prices)


class Strategy:
    """
    Strategy class is an abstract class that defines the structure of a strategy. Strategies are composed of a list of Insturments, a list of Callable rules, and a list of scalars. The rules are applied to the instruments to generate a DataFrame of positions. The scalars are applied to the positions to generate the final positions DataFrame.
    """

    def __init__(self, risk_target: float, capital: float):
        self.instruments: list[Instrument] = []
        self._risk_target = risk_target
        self._capital = capital
        self.rules: list[Callable] = []
        self.scalars: list[float] = []

    @property
    def positions(self) -> pd.DataFrame:
        if not hasattr(self, "_positions"):
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
        self.rules = [
            partial(risk_parity, std_fn=normal_std, risk_target=risk_target),
            partial(trend_signals, std_fn=normal_std),
            partial(equal_weight),
        ]
        self.scalars = [capital]
        self.fetch_data()  # Fetch the data for the instruments

    def fetch_data(self) -> None:
        """
        The Fetch data method for the Trend Following strategy is requires the following instrument specific data:
        1. Prices(Open, High, Low, Close, Volume)
        2. Backadjusted Prices (Close)
        """
        instrument: Future
        for instrument in self.instruments:
            # Load the front calendar contract data with a daily aggregation
            instrument.add_data(
                schema=Agg.DAILY,
                roll_type=RollType.CALENDAR,
                contract_type=ContractType.FRONT,
            )
        return


def normal_std(prices: pd.Series) -> pd.Series:
    """
    Normal Standard Deviation calculates the standard deviation of the prices using a rolling window of 100 days.
    Args:
    prices (pd.Series): A series of prices

    Returns:
    pd.Series: A series of standard deviations
    """
    # Check if the index is a datetime index
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Index must be a datetime index")
    std: pd.Series = pd.Series(prices.rolling(window=100).std())
    std.dropna(inplace=True)
    assert not std.empty, "Standard Deviation is empty"
    assert isinstance(std.index, pd.DatetimeIndex), "Index must be a datetime index"
    return std


### Rules


def risk_parity(
    instruments: list[Future], std_fn: Callable, risk_target: float
) -> pd.DataFrame:
    instrument: Future
    series_list: list[pd.Series] = []
    for instrument in instruments:
        # WARN: Currently uses the front contract close prices WITHOUT backadjusting for gaps
        series_list.append(
            risk_target
            / std_fn(instrument.get_front().get_close().rename(instrument.get_symbol()))
        )

    df = pd.DataFrame()
    for series in series_list:
        if df.empty:
            df = series.to_frame()
        else:
            df = df.join(series.to_frame(), how="outer")

    return df


def equal_weight(instruments: list[Future]) -> pd.DataFrame:
    df = pd.DataFrame()
    instrument: Future
    for instrument in instruments:
        if df.empty:
            df = (
                instrument.get_front()
                .get_close()
                .to_frame()
                .rename(columns={"Close": instrument.name})
            )
        else:
            df = df.join(
                instrument.get_front()
                .get_close()
                .to_frame()
                .rename(columns={"Close": instrument.name}),
                how="outer",
            )

    df.ffill(inplace=True)

    not_null_mask = df.notnull()
    weight_mask = 1 / df.notnull().sum(axis=1).astype(int)

    weights = not_null_mask.mul(weight_mask, axis=0)

    return weights


def trend_signals(instruments: list[Future], std_fn: Callable) -> pd.DataFrame:
    forecasts: list[pd.Series] = []
    instrument: Future
    for instrument in instruments:
        trend = pd.DataFrame()

        crossovers = [(2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256)]

        # Calculate the exponential moving averages crossovers and store them in the trend dataframe for t1, t2 in crossovers: trend[f"{t1}-{t2}"] = data["Close"].ewm(span=t1, min_periods=2).mean() - data["Close"].ewm(span=t2, min_periods=2).mean()
        for t1, t2 in crossovers:
            trend[f"{t1}-{t2}"] = (
                instrument.get_front()
                .get_backadjusted()
                .ewm(span=t1, min_periods=2, adjust=False)
                .mean()
                - instrument.get_front()
                .get_backadjusted()
                .ewm(span=t2, min_periods=2, adjust=False)
                .mean()
            )

        # Calculate the risk adjusted forecasts
        for t1, t2 in crossovers:
            trend[f"{t1}-{t2}"] /= std_fn(
                instrument.get_front()
                .get_backadjusted()
                .rename(instrument.get_symbol())
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
            df = df.join(series.to_frame(), how="outer")

    return df


### Main Function (Implementation, Specific to the Example)


def main():
    instruments: list[Future] = [Future(symbol="ES", dataset="CME", multiplier=50)]
    trend_following: TrendFollowing = TrendFollowing(instruments, 0.2, 100_000)
    positions: pd.DataFrame = trend_following.positions
    print(positions)


if __name__ == "__main__":
    main()
