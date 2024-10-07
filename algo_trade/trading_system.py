from abc import ABC
from decimal import Decimal
from dotenv import load_dotenv
from typing import TypeVar, Generic, Callable, Optional

import pandas as pd # type: ignore

# Internal
from algo_trade.instrument import Instrument, SecurityType
from algo_trade.pnl import PnL
from algo_trade.risk_measures import RiskMeasure
from algo_trade.strategy import Strategy
from algo_trade.ib_utils.account import Account, Position
from algo_trade.ib_utils._contract import Contract

load_dotenv()

T = TypeVar('T', bound=Instrument)

class TradingSystem(ABC, Generic[T]):
    def __init__(
            self,
            instruments : Optional[list[T]] = None,
            weighted_strategies : Optional[list[tuple[float, Strategy]]] = None,
            capital : Optional[float] = None,
            risk_object : Optional[RiskMeasure] = None,
            trading_system_rules : Optional[list[Callable]] = None
        ) -> None:

        self.instruments : Optional[list[T]] = instruments
        self.weighted_strategies : Optional[list[tuple[float, Strategy]]] = weighted_strategies
        self.capital : Optional[float] = capital
        self.risk_object : Optional[RiskMeasure] = risk_object
        self.trading_system_rules : Optional[list[Callable]] = trading_system_rules

    @property
    def security_types(self) -> pd.DataFrame: 
        #* existence of this method, is likely a risk in & of itself, 
        #* because TradingSystems should be homogenous instrument types
        if not hasattr(self, '_security_types'):
            if self.instruments is None:
                raise ValueError("No instruments in the TradingSystem")
            
            security_types : dict[str, SecurityType] = {}
            for instrument in self.instruments:
                security_types[instrument.name] = instrument.security_type

            self._security_types : pd.DataFrame = pd.DataFrame(
                security_types, index=[0], dtype=object)

        return self._security_types

    @property
    def multipliers(self) -> pd.DataFrame:
        if not hasattr(self, '_multipliers'):
            if self.instruments is None:
                raise ValueError("No instruments in the TradingSystem")

            multipliers : dict[str, float] = {}
            for instrument in self.instruments:
                multipliers[instrument.name] = instrument.multiplier

            self._multipliers = pd.DataFrame(multipliers, index=[0], dtype=float)
        
        return self._multipliers

    @property
    def exchanges(self) -> pd.DataFrame:
        if not hasattr(self, '_exchanges'):
            if self.instruments is None:
                raise ValueError("No instruments in the TradingSystem")

            exchanges : dict[str, str] = {}
            for instrument in self.instruments:
                exchanges[instrument.name] = instrument.exchange

            self._exchanges = pd.DataFrame(exchanges, index=[0], dtype=str)

        return self._exchanges
    
    @property
    def currencies(self) -> pd.DataFrame:
        if not hasattr(self, '_currencies'):
            if self.instruments is None:
                raise ValueError("No instruments in the TradingSystem")

            currencies : dict[str, str] = {}
            for instrument in self.instruments:
                currencies[instrument.name] = instrument.currency

            self._currencies = pd.DataFrame(currencies, index=[0], dtype=str)

        return self._currencies

    @property
    def prices(self) -> pd.DataFrame:
        if not hasattr(self, '_prices'):
            if self.instruments is None:
                raise ValueError("No instruments in the TradingSystem")

            self._prices : pd.DataFrame = pd.DataFrame(dtype=float)
            for instrument in self.instruments:
                if self._prices.empty:
                    self._prices = instrument.price.to_frame(instrument.name)
                else:
                    self._prices = self._prices.join(
                        instrument.price.to_frame(instrument.name),
                        how='outer'
                    )

        return self._prices

    @property
    def positions(self) -> pd.DataFrame:
        if not hasattr(self, '_positions'):
            if self.weighted_strategies is None:
                raise ValueError("No strategies in the TradingSystem")

            self._positions : pd.DataFrame = pd.DataFrame(dtype=float)
            for weight, strategy in self.weighted_strategies:
                df : pd.DataFrame = strategy.positions * weight
                self._positions = df if self._positions.empty else self._positions + df

            self._positions /= self.multipliers.iloc[0] # Divide by multipliers

            if self.trading_system_rules is not None:
                for rule in self.trading_system_rules:
                    self._positions = rule(self)

        return self._positions
    
    @positions.setter
    def positions(self, value : pd.DataFrame) -> None:
        if not isinstance(value, pd.DataFrame):
            raise ValueError("Positions must be a pandas DataFrame")
        self._positions = value

    @property
    def exposure(self) -> pd.DataFrame:
        return self.positions * self.prices * self.multipliers.iloc[0]

    @property
    def PnL(self) -> PnL:
        if self.capital is None:
            raise ValueError("No capital in the TradingSystem")

        return PnL(self.positions, self.prices, self.capital, self.multipliers)

    def __getitem__(self, key : int) -> Account:
        positions : pd.DataFrame = self.positions.iloc[[key]]

        ibkr_positions : list[Position] = [
            Position(
                Contract.from_instrument(instrument),
                Decimal(int(positions[instrument.symbol].iloc[0])))
            for instrument in self.instruments
        ]

        return Account(ibkr_positions)

    def __sub__(self, other : 'TradingSystem') -> 'TradingSystem':
        if not isinstance(other, TradingSystem):
            raise ValueError("Can only subtract TradingSystem from TradingSystem")

        self.positions = self.positions - other.positions
        return self
    
    def __add__(self, other : 'TradingSystem') -> 'TradingSystem':
        if not isinstance(other, TradingSystem):
            raise ValueError("Can only add TradingSystem to TradingSystem")

        self.positions = self.positions + other.positions
        return self
