import pandas as pd
import numpy as np
from abc import ABC

from typing import Callable
from functools import partial

from algo_trade.instrument import Future, Instrument, RollType, ContractType, Agg
from algo_trade.risk_measures import RiskMeasure, GARCH
from algo_trade.rules import capital_scaling, risk_parity, equal_weight, trend_signals

class Strategy(ABC):
    """
    Strategy class is an abstract class that defines the structure of a strategy. Strategies are composed of a list of Insturments, a list of Callable rules, and a list of scalars. The rules are applied to the instruments to generate a DataFrame of positions. The scalars are applied to the positions to generate the final positions DataFrame.
    """

    def __init__(self, capital: float):
        self.instruments: list[Instrument] = []
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
        super().__init__(capital=capital)
        # Overload the instruments
        self.instruments: list[Future] = instruments
        self.risk_object = GARCH(
            risk_target=risk_target,
            instruments=instruments,
            weights=(0.01, 0.01, 0.98),
            minimum_observations=100
        )
        
        self.rules = [
            partial(risk_parity, risk_object=self.risk_object),
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
