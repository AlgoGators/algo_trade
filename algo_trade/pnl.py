from enum import Enum, auto
import pandas as pd
import numpy as np
from algo_trade._constants import DAYS_IN_YEAR

class PnL:
    class ReturnType(Enum):
        POINT = auto()
        PERCENT = auto()
    class Timespan(Enum):
        DAILY = auto()
        ANNUALIZED = auto()
        CUMULATIVE = auto()

    def __init__(self, positions : pd.DataFrame, prices : pd.DataFrame, capital : float, multipliers : pd.DataFrame | None = None):
        self.__positions = positions
        self.__prices = prices
        self.__multipliers = multipliers if multipliers is not None else pd.DataFrame(columns=positions.columns, data=np.ones((1, len(positions.columns))))
        self.__capital = capital
        self.__point_returns = self.__get_point_returns()
    
    def get(self, return_type : ReturnType, timespan : Timespan, aggregate : bool = True) -> pd.DataFrame:
        match return_type, timespan:
            case self.ReturnType.POINT, self.Timespan.DAILY:
                return self.__point_returns.sum(axis=1) if aggregate else self.__point_returns
            # case self.ReturnType.POINT, self.Timespan.ANNUALIZED:
            #     return (self.__point_returns * DAYS_IN_YEAR).sum(axis=1) if aggregate else self.__point_returns
            case self.ReturnType.POINT, self.Timespan.CUMULATIVE:
                return self.__point_returns.cumsum().sum(axis=1) if aggregate else self.__point_returns.cumsum()

            case self.ReturnType.PERCENT, self.Timespan.DAILY:
                return self.__point_returns / self.__prices.shift(1) if not aggregate else self.__portfolio_percent_returns(self.__capital)
            # case self.ReturnType.PERCENT, self.Timespan.ANNUALIZED:
            #     return self.__point_returns / self.__prices.shift(1) * DAYS_IN_YEAR if not aggregate else self.__portfolio_percent_returns(self.__capital) * DAYS_IN_YEAR
            case self.ReturnType.PERCENT, self.Timespan.CUMULATIVE:
                return (self.__point_returns / self.__prices.shift(1) + 1).cumprod() - 1 if not aggregate else self.__point_returns.sum(axis=1).cumsum() / self.__capital

            case _:
                raise NotImplementedError(f"The Enums provided or the combination of them: {return_type, timespan}, has not been implemented.")

    def get_sharpe_ratio(self, aggregate : bool = True) -> pd.Series:
        returns = self.get(self.ReturnType.PERCENT, self.Timespan.DAILY, aggregate)
        return returns.mean() / returns.std() * DAYS_IN_YEAR ** 0.5

    def get_volatility(self, timespan : Timespan, aggregate : bool = True) -> pd.Series:
        returns = self.get(self.ReturnType.PERCENT, self.Timespan.DAILY, aggregate)
        if timespan == self.Timespan.DAILY:
            return returns.std() 
        elif timespan == self.Timespan.ANNUALIZED:
            return returns.std() * DAYS_IN_YEAR ** 0.5
        else: 
            raise NotImplementedError(f"The Enum provided: {timespan}, has not been implemented.")

    def get_mean_return(self, timespan : Timespan, aggregate : bool = True) -> pd.Series:
        if timespan == self.Timespan.DAILY:
            returns = self.get(self.ReturnType.PERCENT, self.Timespan.DAILY, aggregate)
            return returns.mean()
        elif timespan == self.Timespan.CUMULATIVE:
            returns = self.get(self.ReturnType.PERCENT, self.Timespan.CUMULATIVE, aggregate)
            total_return = returns.iloc[-1]
            cagr = (1 + total_return) ** (1 / (returns.count() / DAYS_IN_YEAR)) - 1
            return cagr
        raise NotImplementedError(f"The Enum provided: {timespan}, has not been implemented.")

    def __portfolio_percent_returns(self, capital : float) -> pd.Series:
        capital_series = pd.Series(data=capital, index=self.__point_returns.index) + self.__point_returns.sum(axis=1)
        return capital_series / capital_series.shift(1) - 1

    def __get_point_returns(self) -> pd.DataFrame:
        pnl = pd.DataFrame()
        for instrument in self.__positions.columns:
            pos_series = self.__positions[instrument].groupby(self.__positions[instrument].index).last()
            both_series = pd.concat([pos_series, self.__prices[instrument]], axis=1)
            both_series.columns = ["positions", "prices"]
            both_series = both_series.ffill()

            price_returns = both_series.prices.diff()

            returns = both_series.positions.shift(1) * price_returns

            returns[returns.isna()] = 0.0

            pnl[instrument] = returns * self.__multipliers[instrument].iloc[0]

        return pnl