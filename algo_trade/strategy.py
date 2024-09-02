import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from typing import Callable, Generic, TypeVar

from algo_trade.instrument import Instrument, Future, RollType, ContractType, Agg
from algo_trade.risk_measures import RiskMeasure

T = TypeVar('T', bound='Instrument')

class Strategy(Generic[T], ABC):
    """
    Strategy class is an abstract class that defines the structure of a strategy. Strategies are composed of a list of Insturments, a list of Callable rules, and a list of scalars. The rules are applied to the instruments to generate a DataFrame of positions. The scalars are applied to the positions to generate the final positions DataFrame.
    """

    def __init__(self, capital: float):
        self.instruments: list[T] = []
        self._capital = capital
        self.risk_object : RiskMeasure = None
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

    @abstractmethod
    def fetch_data(self):
        """
        The Fetch data method is the a required initialization step within designing a strategy. This method is used to fetch the data for the instruments within the strategy. It is strategy specific and should be implemented by the user.
        """
        pass

class FutureDataFetcher:
    @staticmethod
    def fetch_front(instruments : list[Future]) -> None:
        for instrument in instruments:
            instrument.add_data(Agg.DAILY, RollType.CALENDAR, ContractType.FRONT)
    
    @staticmethod
    def fetch_back(instruments : list[Future]) -> None:
        for instrument in instruments:
            instrument.add_data(Agg.DAILY, RollType.CALENDAR, ContractType.BACK)