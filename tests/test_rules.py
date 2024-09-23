import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd # type: ignore

from algo_trade.instrument import Future
from algo_trade.risk_measures import RiskMeasure, StandardDeviation
from algo_trade.rules import capital_scaling, equal_weight, IDM, risk_parity, trend_signals

class TestRiskManagementFunctions(unittest.TestCase):

    @patch('algo_trade.instrument.Future')
    def test_capital_scaling(self, MockFuture):
        # Setup
        mock_instrument = MockFuture.return_value
        mock_instrument.name = 'Instrument1'  # Set the instrument name
        mock_instrument.front.get_close.return_value = pd.Series([100, 200, 300], index=pd.date_range('2023-01-01', periods=3))
        instruments = [mock_instrument]
        capital = 1000

        # Execution
        result = capital_scaling(instruments, capital)

        # Expected output
        expected = pd.DataFrame({'Instrument1': [10.0, 5.0, 10/3]}, index=pd.date_range('2023-01-01', periods=3))

        # Assertion
        pd.testing.assert_frame_equal(result, expected, atol=1e-6)

    @patch('algo_trade.risk_measures.StandardDeviation')
    @patch('algo_trade.risk_measures.RiskMeasure')
    def test_risk_parity(self, MockRiskMeasure, MockStandardDeviation):
        # Setup
        mock_risk_object = MockRiskMeasure.return_value
        mock_std = MockStandardDeviation.return_value

        mock_risk_object.get_var.return_value.to_standard_deviation.return_value = StandardDeviation(
            data=pd.DataFrame(data={'instrument1': [0.20 / 16]}, index=[0], dtype=np.float64))
        
        # Mocking the annualize method
        mock_std.annualize.return_value = None

        mock_risk_object.tau = 0.20

        # Execution
        result = risk_parity(mock_risk_object)

        # Expected output
        expected = pd.DataFrame(data={'instrument1': 1.0}, index=[0], dtype=np.float64)
        pd.testing.assert_frame_equal(result, expected)

    @patch('algo_trade.instrument.Future')
    def test_equal_weight(self, MockFuture):
        # Setup
        mock_instrument = MockFuture.return_value
        mock_instrument.front.get_close.return_value = pd.Series([100, 200, 300], index=pd.date_range('2023-01-01', periods=3))
        instruments = [mock_instrument]

        # Execution
        result = equal_weight(instruments)

        # Expected output
        expected = pd.DataFrame({mock_instrument.name: [1.0/1, 1.0/1, 1.0/1]}, index=pd.date_range('2023-01-01', periods=3))
        pd.testing.assert_frame_equal(result, expected)

    @patch('algo_trade.instrument.Future')
    @patch('algo_trade.risk_measures.RiskMeasure')
    def test_trend_signals(self, MockRiskMeasure, MockFuture):
        # Setup
        mock_instrument = MockFuture.return_value
        mock_instrument.front.backadjusted = pd.Series([100, 110, 120], index=pd.date_range('2023-01-01', periods=3))
        mock_instrument.front.close = pd.Series([100, 110, 120], index=pd.date_range('2023-01-01', periods=3))
        mock_instrument.get_symbol.return_value = 'instrument1'  # Return a symbol that can be used for lookup in std
        mock_instrument.name = 'instrument1'  # Set a name for the mock instrument

        mock_risk_object = MockRiskMeasure.return_value
        # Return a proper StandardDeviation object for get_var().to_standard_deviation()
        mock_std = MagicMock(spec=StandardDeviation)
        mock_std.__getitem__.side_effect = lambda x: 0.1 if x == 'instrument1' else None
        mock_risk_object.get_var.return_value.to_standard_deviation.return_value = mock_std

        # Mocking the annualize method
        mock_std.annualize.return_value = None  # Simulate inplace=True behavior

        instruments = [mock_instrument]

        # Execution
        result = trend_signals(instruments, mock_risk_object)

        # Expected output
        expected = pd.DataFrame(index=pd.date_range('2023-01-01', periods=3), columns=[mock_instrument.name])
        expected[mock_instrument.name] = [np.nan, 0.7875426999974217, 1.9124573000025786]  # Adjust based on your calculations

        pd.testing.assert_frame_equal(result, expected, atol=1e-6)

    @patch('algo_trade.risk_measures.RiskMeasure')
    @patch('algo_trade.rules.equal_weight')
    def test_IDM(self, mock_equal_weight, MockRiskMeasure):
        # Setup
        mock_risk_object = MockRiskMeasure.return_value
        mock_risk_object.get_returns.return_value = pd.DataFrame({
            'instrument1': [0.01, 0.02, 0.03],
            'instrument2': [0.01, 0.02, 0.03]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        mock_risk_object.instruments = [MagicMock(name='instrument1'), MagicMock(name='instrument2')]
        for i, instrument in enumerate(mock_risk_object.instruments):
            instrument.name = f'instrument{i+1}'

        # Mock equal_weight function
        mock_equal_weight.return_value = pd.DataFrame({
            'instrument1': [0.5, 0.5, 0.5],
            'instrument2': [0.5, 0.5, 0.5]
        }, index=pd.date_range('2023-01-01', periods=3))

        # Mock the rolling correlation matrix
        mock_corr_matrix = pd.DataFrame(
            [[1.0, 0.25], [0.25, 1.0]],
            index=['instrument1', 'instrument2'],
            columns=['instrument1', 'instrument2']
        )
        with patch('pandas.DataFrame.rolling') as mock_rolling:
            mock_rolling.return_value.corr.return_value = pd.concat([mock_corr_matrix]*3, keys=pd.date_range('2023-01-01', periods=3))

            # Execution
            result = IDM(mock_risk_object)

            # Expected output
            expected = pd.DataFrame({
                'instrument1': [1.2649110640673518]*3,
                'instrument2': [1.2649110640673518]*3
            }, index=pd.date_range('2023-01-01', periods=3))

            pd.testing.assert_frame_equal(result, expected, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
