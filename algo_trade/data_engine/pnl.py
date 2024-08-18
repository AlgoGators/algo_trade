from enum import Enum, auto
import pandas as pd
import numpy as np

DAYS_IN_YEAR = 256

### PnL

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
    
    def get(self, return_type : ReturnType, timespan : Timespan, aggregate : bool = False) -> pd.DataFrame:
        match return_type, timespan:
            case self.ReturnType.POINT, self.Timespan.DAILY:
                return self.__point_returns.sum(axis=1) if aggregate else self.__point_returns
            case self.ReturnType.POINT, self.Timespan.ANNUALIZED:
                return (self.__point_returns * DAYS_IN_YEAR).sum(axis=1) if aggregate else self.__point_returns
            case self.ReturnType.POINT, self.Timespan.CUMULATIVE:
                return self.__point_returns.cumsum().sum(axis=1) if aggregate else self.__point_returns.cumsum()

            case self.ReturnType.PERCENT, self.Timespan.DAILY:
                return self.__point_returns / self.__prices.shift(1) if not aggregate else self.__portfolio_percent_returns(self.__capital)
            case self.ReturnType.PERCENT, self.Timespan.ANNUALIZED:
                return self.__point_returns / self.__prices.shift(1) * DAYS_IN_YEAR if not aggregate else self.__portfolio_percent_returns(self.__capital) * DAYS_IN_YEAR
            case self.ReturnType.PERCENT, self.Timespan.CUMULATIVE:
                return (self.__point_returns / self.__prices.shift(1) + 1).cumprod() - 1 if not aggregate else self.__point_returns.sum(axis=1).cumsum() / self.__capital

            case _:
                raise NotImplementedError

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