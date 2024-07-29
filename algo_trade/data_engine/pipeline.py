# Pipeline: The pipeline kicks off the updating and downloading of the futures data, the transformation of the data, and storage of the data

import os

import pandas as pd

from .portfolio import HistoricalPortfolio, LivePortfolio
from .transformation import Transforms

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

    def signals(self):
        return self.t.signals()
    
    def get_trend_tables(self) -> dict[str, pd.DataFrame]:
        return self.t.get_trend_tables()

    """
    We need a JSON Doc of a critical pieces of data.
    RISK MEASURES:
    The first is a dictionary of dataframes where the key is the instrument name and the value is a dataframe of the trend tables. We need to be able to access this at data['trend_tables']
    We well be returned a dictionary of different risk measures we need to be able to connect to the dyn opt

    DYNAMIC OPTIMIZATION:
    """
