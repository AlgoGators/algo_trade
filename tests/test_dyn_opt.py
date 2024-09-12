import unittest
from functools import partial
import pandas as pd
import numpy as np

from algo_trade.dyn_opt import dyn_opt, single_day_optimization
from algo_trade.contract import Contract
from algo_trade.portfolio import Portfolio
from algo_trade.instrument import Future, Agg
from algo_trade.risk_measures import GARCH, RiskMeasure, Covariance
from algo_trade.strategy import Strategy
from algo_trade.rules import capital_scaling, equal_weight, risk_parity

from tests.utils import PriceSeries

def mockedPortfolioMultiplier(*args, **kwargs) -> float:
    return 1.0

def mockedPositionLimit(*args, **kwargs) -> np.ndarray:
    return args[1]

class SimRiskObject():
    def __init__(self, *args, **kwargs):
        self.returns = None
        self.product_returns = None
        self.var = None
        self.covar = None
        self.jump_covar = None
        self.tau = None

    def get_returns(self):
        return self.returns

    def get_product_returns(self):
        return self.product_returns

    def get_var(self):
        return self.var

    def get_cov(self):
        return self.covar

    def get_jump_cov(self, *args, **kwargs):
        return self.jump_covar

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
            if hasattr(instrument, "_price"):
                continue
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
            position_limit_fn=partial(mockedPositionLimit),
            portfolio_multiplier_fn=partial(mockedPortfolioMultiplier)
        )

        #* Using median because sometimes round might go up or down by 1 where dyn_opt might round slightly differently
        #* Median of 0 means its almost they are substantially the same for all SEEDs
        self.assertTrue((optimized_positions - portfolio.positions)[1:].median().median()==0) 
    
    def test_single_day_no_risk(self):
        capital = 500_000
        tau = 0.20
        asymmetric_risk_buffer = 0.05
        unadj_prices = pd.read_parquet('tests/testing_data/dyn_opt/unadj_prices.parquet')[-500:]
        multipliers = pd.read_parquet('tests/testing_data/dyn_opt/multipliers.parquet')[-500:]
        ideal_positions = pd.read_parquet('tests/testing_data/dyn_opt/ideal_positions.parquet')[-500:]
        covariances = pd.read_parquet('tests/testing_data/dyn_opt/covariances.parquet')[-500:]
        jump_covariances = pd.read_parquet('tests/testing_data/dyn_opt/jump_covariances.parquet')[-500:]
        volume = pd.read_parquet('tests/testing_data/dyn_opt/open_interest.parquet')[-500:]
        held_positions = pd.read_parquet('tests/testing_data/dyn_opt/optimized_positions.parquet')[-500:]

        covar = Covariance().from_frame(covariances)
        jump_covar = Covariance().from_frame(jump_covariances)

        notional_exposure_per_contract = unadj_prices * multipliers.iloc[0]
        weight_per_contract = notional_exposure_per_contract / capital

        fixed_cost_per_contract = 3.0
        one_day_costs = np.array(fixed_cost_per_contract)
        instrument_weight = pd.DataFrame(1 / len(ideal_positions.columns), index=ideal_positions.index, columns=ideal_positions.columns)
        cost_penalty_scalar = 10

        # only use last 500 positions (everything else should take care of itself in the code)
        ideal_positions = ideal_positions[-500:]

        optimized_positions : np.ndarray = single_day_optimization(
            held_positions_one_day=held_positions.iloc[-2].values, # -2 because we want to use the previous day's positions
            ideal_positions_one_day=ideal_positions.iloc[-1].values,
            costs_per_contract_one_day=one_day_costs,
            weight_per_contract_one_day=weight_per_contract.iloc[-1].values,
            instrument_weight_one_day=instrument_weight.iloc[-1].values,
            notional_exposure_per_contract_one_day=notional_exposure_per_contract.iloc[-1].values,
            covariances_one_day=covar.iloc[-1].values,
            jump_covariances_one_day=jump_covar.iloc[-1].values,
            volume_one_day=volume.iloc[-1].values,
            tau=tau,
            capital=capital,
            asymmetric_risk_buffer=asymmetric_risk_buffer,
            cost_penalty_scalar=cost_penalty_scalar,
            additional_data=(ideal_positions.columns, ideal_positions.index[-1]),
            optimization=True,
            position_limit_fn=partial(mockedPositionLimit),
            portfolio_multiplier_fn=partial(mockedPortfolioMultiplier)
        )

        expected_df = pd.read_parquet('tests/testing_data/dyn_opt/optimized_positions.parquet')
        expected_df.index = pd.to_datetime(expected_df.index)

        np.testing.assert_array_almost_equal(optimized_positions, expected_df.iloc[-1].values, decimal=0)
        
    def test_in_aggregate(self):
        capital = 500_000
        tau = 0.20
        asymmetric_risk_buffer = 0.05
        unadj_prices = pd.read_parquet('tests/testing_data/dyn_opt/unadj_prices.parquet')[-500:]
        multipliers = pd.read_parquet('tests/testing_data/dyn_opt/multipliers.parquet')[-500:]
        ideal_positions = pd.read_parquet('tests/testing_data/dyn_opt/ideal_positions.parquet')[-500:]
        covariances = pd.read_parquet('tests/testing_data/dyn_opt/covariances.parquet')[-500:]
        jump_covariances = pd.read_parquet('tests/testing_data/dyn_opt/jump_covariances.parquet')[-500:]
        volume = pd.read_parquet('tests/testing_data/dyn_opt/open_interest.parquet')[-500:]

        # only use last 500 positions (everything else should take care of itself in the code)
        ideal_positions = ideal_positions[-500:]

        instruments : list[Future] = []

        for column in ideal_positions.columns:
            future = Future(symbol=column, dataset="CME", multiplier=multipliers[column].iloc[0])
            contract = Contract(future.name, future.dataset, Agg.DAILY)
            contract.volume = volume[column]
            contract.close = unadj_prices[column]
            future.front = contract
            future.price = contract.close
            instruments.append(future)

        portfolio : Portfolio = TestPortfolio(instruments=instruments, risk_target=tau, capital=capital)

        portfolio.positions = ideal_positions
        portfolio.positions.index = pd.to_datetime(portfolio.positions.index)
        sim_risk_object = SimRiskObject()
        sim_risk_object.covar = Covariance().from_frame(covariances)
        sim_risk_object.jump_covar = Covariance().from_frame(jump_covariances)
        sim_risk_object.tau = tau
        portfolio.risk_object = sim_risk_object

        df = dyn_opt(
            portfolio=portfolio,
            instrument_weights=equal_weight(instruments=instruments),
            cost_per_contract=3.0,
            asymmetric_risk_buffer=asymmetric_risk_buffer,
            cost_penalty_scalar=10,
            position_limit_fn=partial(mockedPositionLimit),
            portfolio_multiplier_fn=partial(mockedPortfolioMultiplier)
        )

        # Only optimized for last 500 values
        expected_df = pd.read_parquet('tests/testing_data/dyn_opt/optimized_positions.parquet')
        expected_df.index = pd.to_datetime(expected_df.index)

        pd.testing.assert_frame_equal(df, expected_df)

    def test_by_hand_no_cost(self):
        one_day_costs = np.zeros(3)
        tau = 0.20
        held_positions_one_day=np.zeros(3)
        ideal_positions_one_day=np.array([1.5, 0.7, -1.1])
        instrument_weight=np.array([1/3, 1/3, 1/3])

        unadj_prices = np.array([100, 200, 300])
        multipliers = np.array([3, 1.5, 1])
        capital = 1_000

        notional_exposure_per_contract = unadj_prices * multipliers
        weight_per_contract = notional_exposure_per_contract / capital
        covariances=np.array([[1.0, 0.5, 0.3],
                              [0.5, 1.0, 0.2],
                              [0.3, 0.2, 1.0]])
        jump_covariances=covariances
        volume=np.array([100, 100, 100])

        asymmetric_risk_buffer = 0.05

        optimized_positions : np.ndarray = single_day_optimization(
            held_positions_one_day=held_positions_one_day,
            ideal_positions_one_day=ideal_positions_one_day,
            costs_per_contract_one_day=one_day_costs,
            weight_per_contract_one_day=weight_per_contract,
            instrument_weight_one_day=instrument_weight,
            notional_exposure_per_contract_one_day=notional_exposure_per_contract,
            covariances_one_day=covariances,
            jump_covariances_one_day=jump_covariances,
            volume_one_day=volume,
            tau=tau,
            capital=capital,
            asymmetric_risk_buffer=asymmetric_risk_buffer,
            cost_penalty_scalar=0,
            additional_data=(['A', 'B', 'C'], '2022-01-01'),
            optimization=True,
            position_limit_fn=partial(mockedPositionLimit),
            portfolio_multiplier_fn=partial(mockedPortfolioMultiplier)
        )

        np.testing.assert_array_equal(optimized_positions, np.array([1, 1, -1]))


if __name__ == "__main__":
    unittest.main(failfast=True)
