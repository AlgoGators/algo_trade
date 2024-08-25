import os
from typing import Any, Dict

import pandas as pd
import toml
from abc import ABC

# Internal
from .strategy import Strategy, TrendFollowing
from algo_trade.instrument import Instrument
from .pnl import PnL
from algo_trade.risk_management.dyn_opt.dyn_opt import aggregator


base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
config_dir = os.path.join(base_dir, "config")
config_path = os.path.join(config_dir, "config.toml")

config: Dict[str, Any] = toml.load(config_path)

class Portfolio(ABC):
    def __init__(self, instruments : list[Instrument], weighted_strategies : list[tuple[float, Strategy]], capital : float, multipliers : pd.DataFrame = None):
        self.instruments = instruments
        self.weighted_strategies = weighted_strategies
        self.capital = capital
        self.multipliers = multipliers if multipliers is not None else pd.DataFrame(columns=[instrument.name for instrument in instruments], data=np.ones((1, len(instruments))))

    @property
    def prices(self):
        if not hasattr(self, '_prices'):
            self._prices = pd.DataFrame()
            for instrument in self.instruments:
                if self._prices.empty:
                    self._prices = instrument.prices.to_frame().rename(columns={'Close': instrument.name})
                else:
                    self._prices = self._prices.join(instrument.prices.to_frame().rename(columns={'Close': instrument.name}), how='outer')

        return self._prices

    @property
    def positions(self) -> pd.DataFrame:
        if not hasattr(self, '_positions'):
            self._positions = pd.DataFrame()
            for weight, strategy in self.weighted_strategies:
                df = strategy.positions * weight
                self._positions = df if self._positions.empty else self._positions + df

        return self._positions
    
    @positions.setter
    def positions(self, value):
        self._positions = value

    @property
    def PnL(self) -> PnL: return PnL(self.positions, self.prices, self.capital, self.multipliers)        

### Example Portfolio
class Trend(Portfolio):
    def __init__(self, instruments : list[Instrument], risk_target : float, capital : float):
        self.strategies = [
            (1.0, TrendFollowing(instruments, risk_target, capital))
        ]
        super().__init__(instruments, self.strategies, capital)

def dynamic_optimization(portfolio : Portfolio):
    instruments = portfolio.instruments
    positions = portfolio.positions
    capital = portfolio.capital

    aggregator(
        portfolio.capital,
        None,
        None,
        None,

    )
