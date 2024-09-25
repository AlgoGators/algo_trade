import unittest
from unittest.mock import MagicMock, patch
from typing import TypeVar, Optional

import numpy as np
import pandas as pd  # type: ignore

from algo_trade.risk_measures import (
    RiskMeasure, Covariance, Variance, Instrument, StandardDeviation, _utils
)
from algo_trade._constants import DAYS_IN_YEAR

T = TypeVar('T', bound='Instrument')

class TestUtils(unittest.TestCase):
    def test_ffill_zero(self):
        # Create a sample DataFrame with NaN and zero values
        data = {
            'A': [np.nan, 0, 2, np.nan, 4],
            'B': [0, 0, np.nan, 3, 0],
            'C': [np.nan, 1, 2, 3, 4]
        }
        df = pd.DataFrame(data, dtype=np.float64)

        original_df = df.copy()  # Save a copy of the original DataFrame

        # Expected DataFrame after applying ffill_zero
        expected_data = {
            'A': [np.nan, 0, 2, 0, 4],
            'B': [0, 0, 0, 3, 0],
            'C': [np.nan, 1, 2, 3, 4]
        }
        expected_df = pd.DataFrame(expected_data, dtype=np.float64)

        result_df = _utils.ffill_zero(df)

        # Check if the result matches the expected DataFrame
        pd.testing.assert_frame_equal(result_df, expected_df)

        pd.testing.assert_frame_equal(df, original_df)  # Ensure original DataFrame is not modified


class TestVariance(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'A': [0.01, 0.02, 0.015],
            'B': [0.005, 0.01, 0.007]
        })
        self.variance = Variance(self.data)

    def test_initialization(self):
        self.assertTrue(isinstance(self.variance, pd.DataFrame))
        pd.testing.assert_frame_equal(self.variance, self.data)
        assert isinstance(self.variance, Variance)

    def test_annualize_not_inplace(self):
        annualized = self.variance.annualize(inplace=False)
        factor = DAYS_IN_YEAR
        expected = self.data * factor
        pd.testing.assert_frame_equal(annualized, expected)
        # Ensure original is not modified
        pd.testing.assert_frame_equal(self.variance, self.data)
        assert isinstance(annualized, Variance)

    def test_annualize_inplace(self):
        self.variance.annualize(inplace=True)
        factor = DAYS_IN_YEAR
        expected = self.data * factor
        pd.testing.assert_frame_equal(self.variance, expected)
        assert isinstance(self.variance, Variance)

    def test_annualize_twice_inplace(self):
        self.variance.annualize(inplace=True)
        first_annualized = self.variance.copy()
        self.variance.annualize(inplace=True)
        # Ensure no change after second annualization
        pd.testing.assert_frame_equal(self.variance, first_annualized)
        assert isinstance(self.variance, Variance)

    def test_to_standard_deviation(self):
        std_dev = self.variance.to_standard_deviation()
        expected = self.data ** 0.5
        pd.testing.assert_frame_equal(std_dev, expected)
        assert isinstance(std_dev, StandardDeviation)

    def test_to_frame(self):
        frame = self.variance.to_frame()
        pd.testing.assert_frame_equal(frame, self.data)
        assert isinstance(frame, pd.DataFrame)

class TestStandardDeviation(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'A': [0.1, 0.2, 0.15],
            'B': [0.05, 0.1, 0.07]
        })
        self.std_dev = StandardDeviation(self.data)

    def test_initialization(self):
        self.assertTrue(isinstance(self.std_dev, pd.DataFrame))
        pd.testing.assert_frame_equal(self.std_dev, self.data)
        assert isinstance(self.std_dev, StandardDeviation)

    def test_annualize_not_inplace(self):
        annualized = self.std_dev.annualize(inplace=False)
        factor = DAYS_IN_YEAR ** 0.5
        expected = self.data * factor
        pd.testing.assert_frame_equal(annualized, expected)
        # Ensure original is not modified
        pd.testing.assert_frame_equal(self.std_dev, self.data)
        assert isinstance(annualized, StandardDeviation)

    def test_annualize_inplace(self):
        self.std_dev.annualize(inplace=True)
        factor = DAYS_IN_YEAR ** 0.5
        expected = self.data * factor
        pd.testing.assert_frame_equal(self.std_dev, expected)
        assert isinstance(self.std_dev, StandardDeviation)

    def test_annualize_twice_inplace(self):
        self.std_dev.annualize(inplace=True)
        first_annualized = self.std_dev.copy()
        self.std_dev.annualize(inplace=True)
        # Ensure no change after second annualization
        pd.testing.assert_frame_equal(self.std_dev, first_annualized)
        assert isinstance(self.std_dev, StandardDeviation)

    def test_to_variance(self):
        variance = self.std_dev.to_variance()
        expected = self.data ** 2
        pd.testing.assert_frame_equal(variance, expected)
        assert isinstance(variance, Variance)

    def test_to_frame(self):
        frame = self.std_dev.to_frame()
        pd.testing.assert_frame_equal(frame, self.data)
        assert isinstance(frame, pd.DataFrame)

class TestRiskMeasure(unittest.TestCase):
    def setUp(self):
        # Set up a mock Instrument with percent_returns and name attributes
        self.instrument1 = MagicMock()
        self.instrument1.percent_returns = pd.Series([0.01, 0.02, -0.01], name="Instrument1")
        self.instrument1.name = "Instrument1"
        
        self.instrument2 = MagicMock()
        self.instrument2.percent_returns = pd.Series([0.03, 0.01, -0.02], name="Instrument2")
        self.instrument2.name = "Instrument2"

        self.instruments = [self.instrument1, self.instrument2]
        
        # Create a subclass of RiskMeasure to test abstract methods
        class MockRiskMeasure(RiskMeasure[T]):
            def __init__(self, tau: Optional[float] = None, fill: bool = False):
                super().__init__(tau)
                self.fill = fill
                self.instruments = []  # Add an instruments attribute

            def get_var(self):
                return MagicMock(spec=Variance)
            
            def get_cov(self):
                mock_cov = MagicMock(spec=Covariance)
                mock_cov.to_frame.return_value = pd.DataFrame(
                    np.random.randn(10, 3),
                    columns=['Instrument1_Instrument1', 'Instrument1_Instrument2', 'Instrument2_Instrument2']
                )
                return mock_cov
            
            def get_product_returns(self):
                returns = [instr.percent_returns for instr in self.instruments]
                product_returns = pd.DataFrame({
                    'Instrument1_Instrument1': returns[0] * returns[0],
                    'Instrument1_Instrument2': returns[0] * returns[1],
                    'Instrument2_Instrument2': returns[1] * returns[1],
                })

                if self.fill:
                    from algo_trade.risk_measures import _utils
                    product_returns = _utils.ffill_zero(product_returns)

                return product_returns

        self.RiskMeasureClass = MockRiskMeasure

    def test_tau_setter_valid(self):
        risk_measure = self.RiskMeasureClass()
        risk_measure.tau = 0.5
        self.assertEqual(risk_measure.tau, 0.5)

    def test_tau_setter_invalid(self):
        risk_measure = self.RiskMeasureClass()
        with self.assertRaises(ValueError):
            risk_measure.tau = -0.1  # Invalid tau (negative)

        with self.assertRaises(TypeError):
            risk_measure.tau = "string"  # Invalid tau (not a float)

    def test_tau_getter_not_set(self):
        risk_measure = self.RiskMeasureClass()
        with self.assertRaises(ValueError):
            _ = risk_measure.tau  # Accessing tau before setting

    def test_get_returns(self):
        risk_measure = self.RiskMeasureClass()
        risk_measure.instruments = self.instruments
        risk_measure.fill = False

        returns = risk_measure.get_returns()
        expected_returns = pd.concat([self.instrument1.percent_returns, self.instrument2.percent_returns], axis=1)
        expected_returns = expected_returns.reindex(sorted(expected_returns.columns), axis=1)

        pd.testing.assert_frame_equal(returns, expected_returns)

    @patch('algo_trade.risk_measures._utils.ffill_zero')
    def test_get_returns_fill(self, mock_ffill_zero):
        risk_measure = self.RiskMeasureClass()
        risk_measure.instruments = self.instruments
        risk_measure.fill = True
        
        # Expecting fill forward to be applied
        risk_measure.get_returns()
        mock_ffill_zero.assert_called_once()

    def test_get_product_returns(self):
        risk_measure = self.RiskMeasureClass()
        risk_measure.instruments = self.instruments
        risk_measure.fill = False

        product_returns = risk_measure.get_product_returns()

        expected_product_returns = pd.DataFrame({
            'Instrument1_Instrument1': self.instrument1.percent_returns * self.instrument1.percent_returns,
            'Instrument1_Instrument2': self.instrument1.percent_returns * self.instrument2.percent_returns,
            'Instrument2_Instrument2': self.instrument2.percent_returns * self.instrument2.percent_returns
        })

        pd.testing.assert_frame_equal(product_returns, expected_product_returns)

    @patch('algo_trade.risk_measures._utils.ffill_zero')
    def test_get_product_returns_fill(self, mock_ffill_zero):
        # Set up the mock to return a DataFrame with expected columns and values
        mock_ffill_zero.return_value = pd.DataFrame({
            'Instrument1_Instrument1': [0.0001, 0.0004],
            'Instrument1_Instrument2': [0.0003, 0.0008],
            'Instrument2_Instrument2': [0.0009, 0.0016]
        })

        risk_measure = self.RiskMeasureClass(tau=0.1, fill=True)
        risk_measure.instruments = self.instruments

        # Call the method
        product_returns = risk_measure.get_product_returns()

        # Check that ffill_zero is called once
        self.assertEqual(mock_ffill_zero.call_count, 1)

        # Validate the result
        expected_columns = ['Instrument1_Instrument1', 'Instrument1_Instrument2', 'Instrument2_Instrument2']
        self.assertListEqual(list(product_returns.columns), expected_columns)

    def test_get_jump_cov_valid(self):
        risk_measure = self.RiskMeasureClass()
        risk_measure.instruments = self.instruments
        risk_measure.fill = False

        cov = risk_measure.get_jump_cov(0.95, window=2)

        self.assertIsInstance(cov, Covariance)

    def test_get_jump_cov_invalid_percentile(self):
        risk_measure = self.RiskMeasureClass()

        with self.assertRaises(ValueError):
            risk_measure.get_jump_cov(1.5, window=2)  # Invalid percentile

    @patch('algo_trade.risk_measures.Covariance.from_frame')
    def test_get_jump_cov_filled(self, mock_covariance_from_frame):
        risk_measure = self.RiskMeasureClass()
        risk_measure.instruments = self.instruments
        risk_measure.fill = True
        
        # Mock the Covariance.from_frame to return an instance of Covariance
        mock_covariance = MagicMock(spec=Covariance)
        mock_covariance_from_frame.return_value = mock_covariance

        risk_measure.get_jump_cov(0.95, window=2)

        mock_covariance_from_frame.assert_called_once()
    
    def test_jump_cov_filled_interpolation(self):
        # Simulate a scenario where interpolation (bfill) occurs
        risk_measure = self.RiskMeasureClass()
        risk_measure.instruments = self.instruments
        risk_measure.fill = True
        
        cov = risk_measure.get_jump_cov(0.95, window=2)
        self.assertTrue(cov.to_frame().notna().all().all())  # Ensure no NaN values

    # Uncomment and complete this test as necessary
    # def test_jump_cov_empty_cached(self):
    #     risk_measure = self.RiskMeasureClass()
    #     risk_measure.instruments = self.instruments
        
    #     # Pre-cache jump covariances
    #     cached_jump_cov = Covariance()
    #     risk_measure._RiskMeasure__jump_covariances = cached_jump_cov
        
    #     cov = risk_measure.get_jump_cov(0.95, window=2)
        
    #     self.assertEqual(cov, cached_jump_cov)  # Ensure cached value is returned

class TestCovariance(unittest.TestCase):
    def setUp(self):
        # Set up a sample covariance matrix, dates, and instrument names
        self.cov_matrices = np.array([
            [[1.0, 0.5], [0.5, 2.0]],
            [[1.1, 0.6], [0.6, 2.1]],
            [[1.2, 0.7], [0.7, 2.2]],
        ])
        self.dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
        self.instrument_names = ['A', 'B']

        # Initialize a Covariance object
        self.cov_obj = Covariance(self.cov_matrices, self.dates, self.instrument_names)

    def test_to_frame(self):
        df = self.cov_obj.to_frame()
        expected_columns = ['A_A', 'A_B', 'B_B']
        self.assertListEqual(df.columns.tolist(), expected_columns)
        self.assertTrue(np.allclose(df.iloc[0].values, [1.0, 0.5, 2.0]))
        self.assertEqual(df.index[0], pd.Timestamp('2023-01-01'))

    def test_to_frame_with_empty_matrices(self):
        empty_cov_obj = Covariance()
        df = empty_cov_obj.to_frame()
        self.assertTrue(df.empty)

    def test_reindex(self):
        new_dates = pd.to_datetime(['2023-01-02', '2023-01-03'])
        self.cov_obj.reindex(new_dates)
        self.assertEqual(len(self.cov_obj._dates), 2)
        self.assertTrue(np.array_equal(self.cov_obj._covariance_matrices, self.cov_matrices[1:]))

    def test_reindex_with_empty(self):
        empty_cov_obj = Covariance()
        with self.assertRaises(ValueError):
            empty_cov_obj.reindex(pd.to_datetime(['2023-01-01']))

    def test_from_frame(self) -> None:
        df = pd.DataFrame({
            'A_A': [1.0, 1.1],
            'A_B': [0.5, 0.6],
            'B_B': [2.0, 2.1],
        }, index=pd.to_datetime(['2023-01-01', '2023-01-02']))
        cov_obj : Covariance = Covariance.from_frame(df)
        self.assertEqual(cov_obj._instrument_names, ['A', 'B'])
        assert cov_obj._covariance_matrices is not None
        self.assertTrue(np.array_equal(cov_obj._covariance_matrices, np.array([
            [[1.0, 0.5], [0.5, 2.0]],
            [[1.1, 0.6], [0.6, 2.1]],
        ])))
        assert cov_obj._dates is not None
        self.assertEqual(cov_obj._dates[0], pd.Timestamp('2023-01-01'))

    def test_iterate(self):
        results = list(self.cov_obj.iterate())
        self.assertEqual(len(results), 3)
        self.assertTrue(np.array_equal(results[0][1], self.cov_matrices[0]))

    def test_dropna(self):
        # Add NaNs to the covariance matrices
        cov_matrices_with_nan = self.cov_matrices.copy()
        cov_matrices_with_nan[1, 0, 1] = np.nan
        cov_obj_with_nan = Covariance(cov_matrices_with_nan, self.dates, self.instrument_names)
        cov_obj_with_nan.dropna()

        # Only rows without NaN should remain
        self.assertEqual(len(cov_obj_with_nan._dates), 2)
        self.assertTrue(np.array_equal(cov_obj_with_nan._covariance_matrices, np.array([
            [[1.0, 0.5], [0.5, 2.0]],
            [[1.2, 0.7], [0.7, 2.2]],
        ])))

    def test_empty_property(self):
        empty_cov_obj = Covariance()
        self.assertTrue(empty_cov_obj.empty)
        self.assertFalse(self.cov_obj.empty)

    def test_iloc(self):
        df = self.cov_obj.iloc[0]
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertTrue(np.array_equal(df.values, self.cov_matrices[0]))

    def test_loc(self):
        df = self.cov_obj.loc[pd.Timestamp('2023-01-02')]
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertTrue(np.array_equal(df.values, self.cov_matrices[1]))

    def test_str_repr(self):
        string_rep = str(self.cov_obj)
        repr_rep = repr(self.cov_obj)
        self.assertIn('A_A', string_rep)
        self.assertIn('A_A', repr_rep)

    def test_getitem(self):
        first_matrix = self.cov_obj[0]
        self.assertTrue(np.array_equal(first_matrix, self.cov_matrices[0]))

    def test_iloc_indexer(self):
        first_matrix_df = self.cov_obj.iloc[0]
        self.assertTrue(np.array_equal(first_matrix_df.values, self.cov_matrices[0]))
        self.assertEqual(first_matrix_df.index.tolist(), self.instrument_names)

    def test_loc_indexer(self):
        second_matrix_df = self.cov_obj.loc[self.dates[1]]
        self.assertTrue(np.array_equal(second_matrix_df.values, self.cov_matrices[1]))
        self.assertEqual(second_matrix_df.index.tolist(), self.instrument_names)

if __name__ == '__main__':
    unittest.main()
