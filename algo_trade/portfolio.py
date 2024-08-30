import os
from typing import Any, Dict, TypeVar, Generic

import pandas as pd
import toml
from abc import ABC
import numpy as np
from typing import Callable

# Internal
from algo_trade.strategy import Strategy
from algo_trade.instrument import Instrument
from algo_trade.pnl import PnL
from algo_trade.risk_measures import RiskMeasure

base_dir = os.path.dirname(os.path.dirname(__file__))
config_dir = os.path.join(base_dir, "config")
config_path = os.path.join(config_dir, "config.toml")

config: Dict[str, Any] = toml.load(config_path)

T = TypeVar('T', bound=Instrument)

class Portfolio(ABC, Generic[T]):
    def __init__(self):
        self.instruments : list[T] = None
        self.weighted_strategies : list[tuple[float, Strategy]]
        self.capital : float
        self.risk_object : RiskMeasure
        self.portfolio_rules : list[Callable]
        self.multipliers : pd.DataFrame

    @property
    def multipliers(self):
        if not hasattr(self, '_multipliers'):
            if self.instruments is None:
                raise ValueError("No instruments in the portfolio")

            self._multipliers = pd.DataFrame(columns=[instrument.name for instrument in self.instruments], data=np.ones((1, len(self.instruments))))
        
        return self._multipliers    

    @property
    def prices(self):
        if not hasattr(self, '_prices'):
            self._prices = pd.DataFrame()
            instrument : Instrument
            for instrument in self.instruments:
                if self._prices.empty:
                    self._prices = instrument.price.to_frame(instrument.name)
                else:
                    self._prices = self._prices.join(instrument.price.to_frame(instrument.name), how='outer')

        return self._prices

    @property
    def positions(self) -> pd.DataFrame:
        if not hasattr(self, '_positions'):
            self._positions = pd.DataFrame()
            for weight, strategy in self.weighted_strategies:
                df = strategy.positions * weight
                self._positions = df if self._positions.empty else self._positions + df

            for rule in self.portfolio_rules:
                self._positions = rule(self)

        return self._positions
    
    @positions.setter
    def positions(self, value):
        self._positions = value

    @property
    def PnL(self) -> PnL: return PnL(self.positions, self.prices, self.capital, self.multipliers)
