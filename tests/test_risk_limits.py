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
        notional_exposure_per_contract = (np.linspace(-1.25, 1, length) * 1000).round()
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

if __name__ == '__main__':
    unittest.main()