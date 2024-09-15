from typing import TypeVar, Generic

import pandas as pd
from abc import ABC
from typing import Callable
from dotenv import load_dotenv
from decimal import Decimal

# Internal
from algo_trade.strategy import Strategy
from algo_trade.instrument import Instrument, SecurityType
from algo_trade.pnl import PnL
from algo_trade.risk_measures import RiskMeasure
from algo_trade.ib_utils.account import Account, Position
from algo_trade.ib_utils._contract import Contract

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
    def security_types(self) -> pd.DataFrame: 
        #* existence of this method, is likely a risk in & of itself, 
        #* because TradingSystems should be homogenous instrument types
        if not hasattr(self, '_security_types'):
            if self.instruments is None:
                raise ValueError("No instruments in the TradingSystem")
            
            security_types = {}
            for instrument in self.instruments:
                security_types[instrument.name] = instrument.security_type

            self._security_types : pd.DataFrame = pd.DataFrame(security_types, index=[0])

        return self._security_types

    @property
    def multipliers(self) -> pd.DataFrame:
        if not hasattr(self, '_multipliers'):
            if self.instruments is None:
                raise ValueError("No instruments in the TradingSystem")

            multipliers = {}
            for instrument in self.instruments:
                multipliers[instrument.name] = instrument.multiplier

            self._multipliers = pd.DataFrame(multipliers, index=[0], dtype=float)
        
        return self._multipliers

    @property
    def exchanges(self) -> pd.DataFrame:
        if not hasattr(self, '_exchanges'):
            if self.instruments is None:
                raise ValueError("No instruments in the TradingSystem")

            exchanges = {}
            for instrument in self.instruments:
                exchanges[instrument.name] = instrument.exchange

            self._exchanges = pd.DataFrame(exchanges, index=[0], dtype=str)

        return self._exchanges
    
    @property
    def currencies(self) -> pd.DataFrame:
        if not hasattr(self, '_currencies'):
            if self.instruments is None:
                raise ValueError("No instruments in the TradingSystem")

            currencies = {}
            for instrument in self.instruments:
                currencies[instrument.name] = instrument.currency

            self._currencies = pd.DataFrame(currencies, index=[0], dtype=str)

        return self._currencies

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
        positions : pd.DataFrame = self.positions.iloc[[key]]
        key_pairs = {instrument.name: instrument.ib_symbol for instrument in self.instruments}

        ibkr_positions : list[Position] = [
            Position(
                Contract(
                    symbol=key_pairs[column],
                    multiplier=self.multipliers[column].iloc[0],
                    exchange=self.exchanges[column].iloc[0],
                    currency=self.currencies[column].iloc[0],
                    secType=self.security_types[column].iloc[0].string
                ),
                Decimal(int(positions[column].iloc[0])))
            for column in positions.columns
        ]

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
