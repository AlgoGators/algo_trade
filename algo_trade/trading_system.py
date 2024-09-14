from typing import TypeVar, Generic

import pandas as pd
from abc import ABC
from typing import Callable
from dotenv import load_dotenv
from decimal import Decimal

# Internal
from algo_trade.strategy import Strategy
from algo_trade.instrument import Instrument
from algo_trade.pnl import PnL
from algo_trade.risk_measures import RiskMeasure
from algo_trade.account import Account
from algo_trade.ib_utils.src._contract import Contract

load_dotenv()

T = TypeVar('T', bound=Instrument)

class TradingSystem(ABC, Generic[T]):
    def __init__(self):
        self.instruments : list[T] = None
        self.weighted_strategies : list[tuple[float, Strategy]]
        self.capital : float
        self.risk_object : RiskMeasure
        self.trading_system_rules : list[Callable] = []

    @property
    def multipliers(self):
        if not hasattr(self, '_multipliers'):
            if self.instruments is None:
                raise ValueError("No instruments in the TradingSystem")

            multipliers = {}
            for instrument in self.instruments:
                multipliers[instrument.name] = instrument.multiplier

            self._multipliers = pd.DataFrame(multipliers, index=[0])
        
        return self._multipliers

    @property
    def prices(self) -> pd.DataFrame:
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

            self._positions /= self.multipliers.iloc[0] # Divide by multipliers

            for rule in self.trading_system_rules:
                self._positions = rule(self)

        return self._positions
    
    @positions.setter
    def positions(self, value):
        self._positions = value

    @property
    def exposure(self) -> pd.DataFrame:
        return self.positions * self.prices * self.multipliers.iloc[0]

    @property
    def PnL(self) -> PnL: return PnL(self.positions, self.prices, self.capital, self.multipliers)

    def __getitem__(self, key) -> Account:
        positions : pd.DataFrame = self.positions.iloc[key]
        key_pairs = {instrument.name: instrument.ib_symbol for instrument in self.instruments}

        ibkr_positions : dict[Contract, Decimal]= {
            Contract(symbol=key_pairs[column], multiplier=self.multipliers[column].iloc[0]): Decimal(int(positions[column].iloc[0]))
            for column in positions.columns
        }

        return Account(ibkr_positions)

    def __sub__(self, other) -> 'TradingSystem':
        if not isinstance(other, TradingSystem):
            raise ValueError("Can only subtract TradingSystem from TradingSystem")
        self.positions = self.positions - other.positions
        return self
    
    def __add__(self, other) -> 'TradingSystem':
        if not isinstance(other, TradingSystem):
            raise ValueError("Can only add TradingSystem to TradingSystem")
        self.positions = self.positions + other.positions
        return self