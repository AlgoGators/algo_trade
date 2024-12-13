import unittest
import pandas as pd # type: ignore
import numpy as np
from algo_trade.pnl import PnL  # Replace with the correct module path for PnL

class TestPnL(unittest.TestCase):
    def setUp(self):
        # Setup some sample data for testing
        self.positions = pd.DataFrame({
            'Instrument_A': [10, 12, 10, 8, 10],
            'Instrument_B': [15, 14, 16, 14, 15]
        }, index=pd.date_range('2023-01-01', periods=5))

        self.prices = pd.DataFrame({
            'Instrument_A': [100, 102, 101, 103, 105],
            'Instrument_B': [200, 198, 202, 199, 201]
        }, index=pd.date_range('2023-01-01', periods=5))

        self.multipliers = pd.DataFrame({
            'Instrument_A': [1],
            'Instrument_B': [1]
        })

        self.capital = 5000  # Capital set to 5000

        # Initialize PnL object
        self.pnl = PnL(self.positions, self.prices, self.capital, self.multipliers)

    def test_get_daily_point_return(self):
        # Test daily point returns without aggregation
        daily_point_return = self.pnl.get(PnL.ReturnType.POINT, PnL.Timespan.DAILY, aggregate=False)
        self.assertIsInstance(daily_point_return, pd.DataFrame)

        # Validate shape and data types
        self.assertEqual(daily_point_return.shape, self.positions.shape)
        self.assertEqual(daily_point_return.sum().sum(), np.float64(50.0))

    def test_get_cumulative_point_return(self):
        # Test cumulative point returns
        cumulative_point_return = self.pnl.get(PnL.ReturnType.POINT, PnL.Timespan.CUMULATIVE, aggregate=False)
        self.assertIsInstance(cumulative_point_return, pd.DataFrame)
        self.assertEqual(cumulative_point_return.iloc[-1].sum(), np.float64(50.0))
        self.assertEqual(cumulative_point_return.shape, self.positions.shape)

    def test_get_daily_percent_return(self):
        # Test daily percent returns without aggregation
        daily_percent_return = self.pnl.get(PnL.ReturnType.PERCENT, PnL.Timespan.DAILY, aggregate=False)
        self.assertIsInstance(daily_percent_return, pd.DataFrame)
        self.assertEqual(daily_percent_return.shape, self.positions.shape)

    def test_get_cumulative_percent_return(self):
        # Test cumulative percent returns
        cumulative_percent_return = self.pnl.get(PnL.ReturnType.PERCENT, PnL.Timespan.CUMULATIVE, aggregate=False)
        self.assertIsInstance(cumulative_percent_return, pd.DataFrame)

    def test_get_sharpe_ratio(self):
        # Test Sharpe Ratio calculation, which should return a float
        sharpe_ratio = self.pnl.get_sharpe_ratio(aggregate=True)
        self.assertIsInstance(sharpe_ratio, float)

    def test_drawdown(self):
        # Test drawdown calculation
        drawdown = self.pnl.drawdown(aggregate=True)
        self.assertIsInstance(drawdown, pd.Series)

    def test_get_max_drawdown(self):
        # Test max drawdown calculation
        max_drawdown = self.pnl.get_max_drawdown(aggregate=True)
        self.assertIsInstance(max_drawdown, float)

    def test_calmar_ratio(self):
        # Test Calmar Ratio calculation
        calmar_ratio = self.pnl.get_calmar_ratio(aggregate=True)
        self.assertIsInstance(calmar_ratio, float)

    def test_tracking_error(self):
        # Test tracking error calculation
        tracking_error = self.pnl.tracking_error(self.prices['Instrument_A'])
        self.assertIsInstance(tracking_error, float)

    def test_information_ratio(self):
        # Test Information Ratio calculation
        information_ratio = self.pnl.get_information_ratio(aggregate=True)
        self.assertIsInstance(information_ratio, float)

    def test_tail_ratio(self):
        # Test Tail Ratio calculation
        tail_ratio = self.pnl.get_tail_ratio(aggregate=True)
        self.assertIsInstance(tail_ratio, float)

    def test_get_skewness(self):
        # Test Skewness calculation
        skewness = self.pnl.get_skewness(aggregate=True)
        self.assertIsInstance(skewness, float)

    def test_get_kurtosis(self):
        # Test Kurtosis calculation
        kurtosis = self.pnl.get_kurtosis(aggregate=True)
        self.assertIsInstance(kurtosis, float)

    def test_turnover(self):
        # Test Turnover calculation
        turnover = self.pnl.get_turnover(aggregate=True)
        self.assertIsInstance(turnover, pd.Series)

    def test_transaction_costs(self):
        # Test Transaction Costs calculation
        transaction_costs = self.pnl.get_transaction_costs(aggregate=True)
        self.assertIsInstance(transaction_costs, pd.Series)

    def test_plot(self):
        # Test Plot functionality
        self.pnl.plot(metrics=['returns', 'cumulative', 'drawdown'], aggregate=True)
        # Note: This test doesn't assert, but ensures no exceptions are raised during plotting

if __name__ == '__main__':
    unittest.main()
