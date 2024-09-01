import unittest
from functools import partial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from algo_trade.dyn_opt import dyn_opt
from algo_trade.contract import Contract
from algo_trade.portfolio import Portfolio
from algo_trade.instrument import Future, RollType, ContractType, Agg
from algo_trade.risk_measures import GARCH, RiskMeasure
from algo_trade.strategy import Strategy
from algo_trade.rules import capital_scaling, equal_weight, risk_parity

from tests.utils import PriceSeries

def portfolioMultiplier(*args, **kwargs) -> float:
    return 1.0

def positionLimit(*args) -> np.ndarray:
    number_of_instruments = len(args[1])
    return np.array([np.inf for _ in range(number_of_instruments)])

SEED = 10

class TestStrategy(Strategy[Future]):
    def __init__(self, instruments: list[Future], risk_object: RiskMeasure, capital: float):
        super().__init__(capital=capital)
        self.instruments: list[Future] = instruments
        self.risk_object = risk_object
        self.rules = [
            partial(equal_weight, instruments=instruments),
            partial(capital_scaling, instruments=instruments, capital=capital),
            partial(risk_parity, risk_object=self.risk_object),
        ]
        self.scalars = []
        self.fetch_data()  # Fetch the data for the instruments

    def fetch_data(self) -> None:
        """
        The Fetch data method for the Trend Following strategy is requires the following instrument specific data:
        1. Prices(Open, High, Low, Close, Volume)
        2. Backadjusted Prices (Close)
        """
        # Load the front calendar contract data with a daily aggregation
        for instrument in self.instruments:
            contract = Contract(instrument.name, instrument.dataset, Agg.DAILY)
            daily_volatility = 0.20 / 16
            daily_return = 0.20 / 256
            duration = 1000
            contract.close = pd.Series(PriceSeries(daily_volatility, daily_return, 100, duration, SEED), index=pd.date_range(start="2022-01-01", periods=duration))
            contract.volume = pd.Series(np.ones(duration)*100, index=pd.date_range(start="2022-01-01", periods=duration))
            instrument.front = contract
            instrument.price = contract.close


class TestPortfolio(Portfolio[Future]):
    def __init__(self, instruments : list[Future], risk_target : float, capital : float):
        super().__init__()
        self.risk_object = GARCH(
            risk_target=risk_target,
            instruments=instruments,
            weights=(0.01, 0.01, 0.98),
            minimum_observations=100
        )
        self.weighted_strategies = [
            (1.0, TestStrategy(instruments, self.risk_object, capital))
        ]
        self.capital = capital
        self.instruments = instruments

class TestDynOpt(unittest.TestCase):
    def test_dyn_opt(self):
        future = Future(symbol="ES", dataset="CME", multiplier=5)

        instruments = [future]

        portfolio = TestPortfolio(instruments, risk_target=0.20, capital=500_000)
        portfolio.positions = portfolio.positions.round()

        optimized_positions = dyn_opt(
            portfolio=portfolio,
            instrument_weights=equal_weight(instruments=instruments),
            cost_per_contract=3.0,
            asymmetric_risk_buffer=0.05,
            cost_penalty_scalar=10,
            position_limit_fn=partial(positionLimit),
            portfolio_multiplier_fn=partial(portfolioMultiplier)
        )

        self.assertTrue((optimized_positions - portfolio.positions)[1:].sum().sum()==0)

    def test_no_adj_needed(self):
        future = Future(symbol="ES", dataset="CME", multiplier=5)

        instruments = [future]

        portfolio = TestPortfolio(instruments, risk_target=0.20, capital=500_000)
        portfolio.positions = portfolio.positions.round()

        optimized_positions = dyn_opt(
            portfolio=portfolio,
            instrument_weights=equal_weight(instruments=instruments),
            cost_per_contract=0.0,
            asymmetric_risk_buffer=0.00,
            cost_penalty_scalar=0,
            position_limit_fn=partial(positionLimit),
            portfolio_multiplier_fn=partial(portfolioMultiplier)
        )

        self.assertTrue((optimized_positions - portfolio.positions)[1:].sum().sum()==0)

if __name__ == "__main__":
    unittest.main()
