# Pipeline: The pipeline kicks off the updating and downloading of the futures data, the transformation of the data, and storage of the data

import os
from typing import Dict

import pandas as pd

from .portfolio import HistoricalPortfolio, LivePortfolio
from .transformation import Transforms

from algo_trade.data_engine.std_daily_price import standardDeviation
from algo_trade.risk_management.risk_measures.risk_measures import RiskMeasures

base_dir = os.path.dirname(__file__)
contracts_path = os.path.join(base_dir, "data", "processed", "contracts.csv")


# * The pipeline class is a helper class
class Pipeline:
    def __init__(self) -> None:
        contracts = pd.read_csv(contracts_path)
        self.symbols = contracts["Data Symbol"].tolist()

    def update(self):
        self.portfolio = LivePortfolio(self.symbols)
        return

    def rebuild(self):
        self.portfolio = HistoricalPortfolio(self.symbols, None, None, full=True)
        return

    def transform(self):
        self.t = Transforms(self.symbols)
        return

    def load(self):
        self.t.load()
        return

    def positions(self, capital: float, tau: float, multipliers: pd.DataFrame, IDM : int, variances : pd.DataFrame) -> pd.DataFrame:
        """
        Positions: This functions calculates the position sizes for each of the instruments in the portfolio.

        Formula(s):
        - position = Capped Combined Signal * Capital * IDM * Weight * tau / (Multiplier * Price * FX * σ %)
        - IDM(Instrument Diversification Multiplier) = 1 / sqrt(w.rho.w.T)
        - rho = Covariance matrix of the instruments in the portfolio ~ NxN
        - w = A vector of the weights of the instruments in the portfolio summing to 1
        - tau = The risk tolerance parameter

        Args:
        - capital: The amount of capital to be allocated to the portfolio 
        - tau: The risk aversion parameter
        - multipliers: The multipliers for each of the instruments in the portfolio

        Returns:
        - A dataframe of the positions for each of the instruments in the portfolio
        """
        w: int = 1 / len(self.symbols)
        signals: pd.DataFrame = self.t.signals(variances)
        price: pd.DataFrame = self.t.get_current_price()
        positions : pd.DataFrame = signals * capital * IDM * w * tau / (variances ** 0.5 * 16) / price / 10
        positions = positions.apply(lambda col: col / multipliers.loc['multiplier', col.name])

        return positions

    def get_price_tables(self) -> dict[str, pd.DataFrame]:
        return self.t.get_price_tables()


    def get_prices(self) -> pd.DataFrame:
        """
        The get_prices function returns the prices of the instruments in the portfolio unadjusted for rollovers

        Args:
        - None

        Returns:
        - A dataframe of the prices of the instruments in the portfolio
        """
        return self.t.get_current_price()

    def get_open_interest(self) -> pd.DataFrame:
        """
        The get_open_interest function returns the open interest of the instruments in the portfolio

        Args:
        - None

        Returns:
        - A dataframe of the open interest of the instruments in the portfolio
        """
        return self.t.get_open_interest()
    """
    We need a JSON Doc of a critical pieces of data.
    RISK MEASURES:
    The first is a dictionary of dataframes where the key is the instrument name and the value is a dataframe of the trend tables. We need to be able to access this at data['trend_tables']
    We well be returned a dictionary of different risk measures we need to be able to connect to the dyn opt

    DYNAMIC OPTIMIZATION:
    """
