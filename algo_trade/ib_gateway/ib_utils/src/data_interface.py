"""Contains the DataInterface class, which is used to convert between IBKR and data positions."""
from decimal import Decimal
import pandas as pd

from ..src._contract import Contract
from ..src._enums import SecurityType

class DataInterface:
    """Interface to convert between IBKR and data positions"""
    def __init__(
        self,
        instruments_df : pd.DataFrame,
        IBKR_positions : dict[Contract, Decimal] = None,
        data_positions : dict[str, int] = None):

        self.instruments_df = instruments_df

        if IBKR_positions and data_positions:
            raise ValueError("Must provide either IBKR_positions or data_positions, not both")
        elif IBKR_positions:
            self._IBKR_positions = IBKR_positions
        elif data_positions:
            self._data_positions = data_positions

        self._data_positions = {} if data_positions is None else data_positions
        self._IBKR_positions = {} if IBKR_positions is None else IBKR_positions

        self.__update_data_positions()
        self.__update_IBKR_positions()

    @property
    def data_positions(self):
        return self._data_positions

    @data_positions.setter
    def set_data_positions(self, data_positions):
        self._data_positions = data_positions
        self.__update_IBKR_positions()

    def __update_data_positions(self):
        if self.IBKR_positions is None or self.IBKR_positions == {}:
            return
        for contract, quantity in self.IBKR_positions.items():
            data_symbol = self.instruments_df.loc[
                self.instruments_df['ibSymbol'] == contract.symbol, 'dataSymbol'].values[0]
            self._data_positions[data_symbol] = int(quantity)

    @property
    def IBKR_positions(self):
        return self._IBKR_positions

    @IBKR_positions.setter
    def IBKR_positions(self, IBKR_positions):
        self._IBKR_positions = IBKR_positions
        self.__update_data_positions()

    def __update_IBKR_positions(self):
        if self.data_positions is None or self.data_positions == {}:
            return
        for symbol, quantity in self.data_positions.items():
            if symbol not in self.instruments_df['dataSymbol'].values:
                raise ValueError(f"Symbol {symbol} not found in instruments_df")

            contract = Contract()
            contract.symbol = self.instruments_df.loc[
                self.instruments_df['dataSymbol'] == symbol, 'ibSymbol'].values[0]
            contract.exchange = self.instruments_df.loc[
                self.instruments_df['dataSymbol'] == symbol, 'exchange'].values[0]
            contract.multiplier = Decimal(self.instruments_df.loc[
                self.instruments_df['dataSymbol'] == symbol, 'multiplier'].values[0])
            contract.currency = self.instruments_df.loc[
                self.instruments_df['dataSymbol'] == symbol, 'currency'].values[0]
            contract.secType = SecurityType.Future
            self._IBKR_positions[contract] = Decimal(quantity)
