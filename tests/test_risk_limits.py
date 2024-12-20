import unittest
import types
import numpy as np
from tests.utils import MockNestedFunction
from enum import Enum

from algo_trade.risk_limits import position_limit, portfolio_multiplier

class RiskLimitFunctions(Enum):
    MAX_LEVERAGE = 'max_leverage'
    CORRELATION_RISK = 'correlation_risk'
    PORTFOLIO_RISK = 'portfolio_risk'
    JUMP_RISK_MULTIPLIER = 'jump_risk_multiplier'
    MAX_FORECAST = 'max_forecast'
    MIN_VOLUME = 'min_volume'

class TestPositionLimits(unittest.TestCase):
    def test_max_leverage(self):
        max_leverage_ratio = 2.0
        max_leverage_fn = MockNestedFunction(
            position_limit, RiskLimitFunctions.MAX_LEVERAGE.value, max_leverage_ratio=max_leverage_ratio)

        capital = 100_000
        notional_exposure_per_contract = (np.linspace(-1, 1, 10) * 1000).round() # evenly spaced from -1000 to 1000

        two_hundred_each = max_leverage_fn(capital=capital, notional_exposure_per_contract=notional_exposure_per_contract)

        np.testing.assert_equal(two_hundred_each, max_leverage_ratio * capital / notional_exposure_per_contract)
    
    def test_max_forecast(self):
        max_forecast_ratio = 2.0
        max_forecast_buffer = 0.5
        IDM = 2.5
        tau = 0.20

        length = 10

        max_forecast_fn = MockNestedFunction(
            position_limit, RiskLimitFunctions.MAX_FORECAST.value, max_forecast_ratio=max_forecast_ratio, max_forecast_buffer=max_forecast_buffer, IDM=IDM, tau=tau)

        capital = 100_000
        notional_exposure_per_contract = (np.linspace(-1, 1, length) * 1000).round() # evenly spaced from -1000 to 1000
        instrument_weight = np.ones(length) * 0.1
        annualized_volatility = np.ones(length) * 0.2

        max_forecast = max_forecast_fn(
            capital=capital, notional_exposure_per_contract=notional_exposure_per_contract, instrument_weight=instrument_weight, annualized_volatility=annualized_volatility)

        np.testing.assert_equal(max_forecast, (1 + max_forecast_buffer) * max_forecast_ratio * capital * IDM * instrument_weight * tau / notional_exposure_per_contract / annualized_volatility)

    def test_min_volume(self):
        minimum_volume = 100
        min_volume_fn = MockNestedFunction(
            position_limit, RiskLimitFunctions.MIN_VOLUME.value, minimum_volume=minimum_volume)

        volume = np.linspace(0, 100, 10).round()

        min_volume : np.ndarray = min_volume_fn(volume)
        
        self.assertEqual(min_volume.sum(), min_volume[-1]) # know only one can be 100

    def test_aggregate(self):
        max_leverage_ratio = 2.0
        max_forecast_ratio = 2.0
        max_forecast_buffer = 0.5
        IDM = 2.5
        tau = 0.20
        minimum_volume = 100

        position_limit_fn = position_limit(
            max_leverage_ratio=max_leverage_ratio,
            max_forecast_ratio=max_forecast_ratio,
            max_forecast_buffer=max_forecast_buffer,
            IDM=IDM,
            tau=tau,
            minimum_volume=minimum_volume)

        length = 10

        capital = 100_000
        notional_exposure_per_contract = (np.linspace(-1.25, 1.0, length) * 1000).round()
        print(notional_exposure_per_contract)
        instrument_weight=np.ones(length) * 0.1
        volume=np.linspace(400, 0, length).round()
        covariance_matrix = np.ones((length, length)) * 0.50 / 16 
        additional_data = ([str(_) for _ in range(length)], '2021-01-01')

        positions = np.ones(length) * 100

        x = position_limit_fn(
            capital=capital,
            positions=positions,
            notional_exposure_per_contract=notional_exposure_per_contract,
            instrument_weight=instrument_weight, 
            covariance_matrix=covariance_matrix,
            volume=volume,
            additional_data=additional_data
        )

        np.testing.assert_equal(x, np.array([24,  30,  40,  60, 100, 100, 100,   0,   0,   0,]))

        positions = np.ones(length) * 10
        volume = np.ones(length) * 100

        x = position_limit_fn(
            capital=capital,
            positions=positions,
            notional_exposure_per_contract=notional_exposure_per_contract,
            instrument_weight=instrument_weight, 
            covariance_matrix=covariance_matrix,
            volume=volume,
            additional_data=additional_data
        )

        np.testing.assert_equal(x, positions)

class TestPortfolioLimits(unittest.TestCase):
    def test_max_leverage(self):
        max_portfolio_leverage = 2.0
        max_leverage_fn = MockNestedFunction(
            portfolio_multiplier, RiskLimitFunctions.MAX_LEVERAGE.value, max_portfolio_leverage=max_portfolio_leverage)

        length = 4

        notional_exposure_per_contract = (np.linspace(-1.25, 1, length) * 1000).round()

        positions = np.linspace(-2, 4, length).round()

        capital = 1_000

        positions_weighted = notional_exposure_per_contract * positions / capital

        max_leverage = max_leverage_fn(positions_weighted)

        self.assertEqual(max_leverage, np.float64(max_portfolio_leverage / 7))

        capital = 100_000

        positions_weighted = notional_exposure_per_contract * positions / capital

        max_leverage = max_leverage_fn(positions_weighted)

        self.assertEqual(max_leverage, np.float64(1.0))

    def test_correlation_risk(self):
        max_correlation_risk = 0.75
        correlation_risk_fn = MockNestedFunction(
            portfolio_multiplier, RiskLimitFunctions.CORRELATION_RISK.value, max_correlation_risk=max_correlation_risk)

        length = 4

        notional_exposure_per_contract = (np.linspace(-1.25, 1, length) * 1000).round()

        positions = np.linspace(-2, 4, length).round()

        capital = 1_000

        positions_weighted = notional_exposure_per_contract * positions / capital

        annualized_volatility = np.ones(length) * 0.2

        correlation_risk = correlation_risk_fn(positions_weighted, np.diag(annualized_volatility))

        self.assertEqual(correlation_risk, np.float64(max_correlation_risk / 1.4))

        annualized_volatility = np.ones(length) * 0.1

        correlation_risk = correlation_risk_fn(positions_weighted, np.diag(annualized_volatility))

        self.assertEqual(correlation_risk, np.float64(1.0))

    def test_portfolio_risk(self):
        max_portfolio_volatility = 0.75
        portfolio_risk_fn = MockNestedFunction(
            portfolio_multiplier, RiskLimitFunctions.PORTFOLIO_RISK.value, max_portfolio_volatility=max_portfolio_volatility)

        length = 4

        notional_exposure_per_contract = (np.linspace(-1.25, 1, length) * 1000).round()

        positions = np.linspace(-2, 4, length).round()

        capital = 1_000

        positions_weighted = notional_exposure_per_contract * positions / capital

        covariance_matrix = np.ones((length, length)) * 0.50 / 8

        portfolio_risk = portfolio_risk_fn(positions_weighted, covariance_matrix)

        self.assertEqual(portfolio_risk, np.float64(max_portfolio_volatility / 1.75))

        covariance_matrix = np.ones((length, length)) * 0.50 / 16

        positions_weighted /= 2

        portfolio_risk = portfolio_risk_fn(positions_weighted, covariance_matrix)

        self.assertEqual(portfolio_risk, np.float64(1.0))

    def test_jump_risk_multiplier(self):
        max_portfolio_jump_risk = 0.75
        jump_risk_multiplier_fn = MockNestedFunction(
            portfolio_multiplier, RiskLimitFunctions.JUMP_RISK_MULTIPLIER.value, max_portfolio_jump_risk=max_portfolio_jump_risk)

        length = 4

        notional_exposure_per_contract = (np.linspace(-1.25, 1, length) * 1000).round()

        positions = np.linspace(-2, 4, length).round()

        capital = 1_000

        positions_weighted = notional_exposure_per_contract * positions / capital

        jump_covariance_matrix = np.ones((length, length)) * 0.50 / 8

        jump_risk_multiplier = jump_risk_multiplier_fn(positions_weighted, jump_covariance_matrix)

        self.assertEqual(jump_risk_multiplier, np.float64(max_portfolio_jump_risk / 1.75))

        jump_covariance_matrix = np.ones((length, length)) * 0.50 / 16

        positions_weighted /= 2

        jump_risk_multiplier = jump_risk_multiplier_fn(positions_weighted, jump_covariance_matrix)

        self.assertEqual(jump_risk_multiplier, np.float64(1.0))

    def test_aggregate(self):
        max_portfolio_leverage = 2.0
        max_correlation_risk = 0.75
        max_portfolio_volatility = 0.75
        max_portfolio_jump_risk = 0.75

        portfolio_multiplier_fn = portfolio_multiplier(
            max_portfolio_leverage=max_portfolio_leverage,
            max_correlation_risk=max_correlation_risk,
            max_portfolio_volatility=max_portfolio_volatility,
            max_portfolio_jump_risk=max_portfolio_jump_risk)

        length = 4

        notional_exposure_per_contract = (np.linspace(-1.25, 1, length) * 1000).round()

        positions = np.linspace(-2, 4, length).round()

        capital = 1_000

        positions_weighted = notional_exposure_per_contract * positions / capital

        covariance_matrix = np.ones((length, length)) * 0.50 / 8
        jump_covariance_matrix = np.ones((length, length)) * 0.50 / 8

        x = portfolio_multiplier_fn(
            positions_weighted=positions_weighted,
            covariance_matrix=covariance_matrix,
            jump_covariance_matrix=jump_covariance_matrix,
            date='2021-01-01'
        )

        self.assertEqual(x, np.float64(0.75 / 1.75))

        capital = 1_000_000

        positions_weighted = notional_exposure_per_contract * positions / capital

        covariance_matrix = np.ones((length, length)) * 0.50 / 16
        jump_covariance_matrix = np.ones((length, length)) * 0.50 / 16

        positions_weighted /= 2

        x = portfolio_multiplier_fn(
            positions_weighted=positions_weighted,
            covariance_matrix=covariance_matrix,
            jump_covariance_matrix=jump_covariance_matrix,
            date='2021-01-01'
        )

        self.assertEqual(x, np.float64(1.0))

if __name__ == '__main__':
    unittest.main(failfast=True)