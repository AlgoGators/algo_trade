"""
Contains Strategy Abstract Base Class and data fetchers (just futures for now)
"""

from abc import ABC, abstractmethod
import asyncio
from typing import Callable, Generic, Optional, TypeVar

import numpy as np
import pandas as pd # type: ignore

from algo_trade.rules import Rule
from algo_trade.instrument import Agg, ContractType, Future, Instrument, RollType
from algo_trade.risk_measures import RiskMeasure

T = TypeVar('T', bound='Instrument')

class Strategy(ABC, Generic[T]):
    """
        Abstract class that defines the structure of a strategy.
        Composed of a list of Instruments, a list of Callable rules, and a list of scalars.
        The rules are applied to the instruments to generate a DataFrame of positions.
        The scalars are applied to the positions to generate the final positions DataFrame.
    """

    def __init__(
            self,
            instruments : Optional[list[T]] = None,
            risk_object : Optional[RiskMeasure] = None,
            rules : Optional[list[Callable]] = None,
            scalars : tuple[np.float64] = (np.float64(1.0),)
        ) -> None:

        self.instruments : Optional[list[T]] = instruments
        self.risk_object : Optional[RiskMeasure[T]] = risk_object
        self.rules : Optional[list[Rule]] = rules
        self.scalars : tuple[np.float64] = scalars

    @property
    def positions(self) -> pd.DataFrame:
        if self.rules is None:
            raise ValueError("Rules must be set before positions can be calculated")

        if not hasattr(self, "_positions"):
            self._positions : pd.DataFrame = pd.DataFrame()
            for rule in self.rules:
                df : pd.DataFrame = rule()
                self._positions = df if self._positions.empty else self._positions * df

            scalar : np.float64 = np.prod(self.scalars)

            self._positions = self._positions * scalar

        return self._positions

    @positions.setter
    def positions(self, value : pd.DataFrame) -> None:
        if not isinstance(value, pd.DataFrame):
            raise ValueError("Positions must be a pandas DataFrame")

        self._positions = value

    @abstractmethod
    async def fetch_data(self) -> None:
        """
            fetch_data method is required when designing a strategy.
            This method is used to fetch the data for the instruments.
            It is strategy specific, depending on instrument and data reqs; MUST be implemented
        """

class FutureDataFetcher:
    @staticmethod
    async def fetch_front(instruments: list[Future]) -> None:
        """
            Fetches the front contract data for the instruments
        """
        await asyncio.gather(*[
            instrument.add_data(Agg.DAILY, RollType.CALENDAR, ContractType.FRONT)
            for instrument in instruments
        ])

    @staticmethod
    async def fetch_back(instruments: list[Future]) -> None:
        """
            Fetches the back contract data for the instruments
        """
        await asyncio.gather(*[
            instrument.add_data(Agg.DAILY, RollType.CALENDAR, ContractType.BACK)
            for instrument in instruments
        ])
