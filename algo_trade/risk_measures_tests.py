import pandas as pd
import numpy as np

from risk_measures import StandardDeviation, Variance, _utils


def generate_mock_data():
    # Create a mock DataFrame with random returns for testing
    dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
    data = np.random.randn(100, 2)  # 100 rows, 2 columns of random data
    df = pd.DataFrame(data, index=dates, columns=['Asset1', 'Asset2'])
    return df


def test_ffill_zero():
    # Test for the forward fill zero utility function
    df = generate_mock_data()
    df.iloc[5:10] = 0  # Introduce some zeros in the data
    filled_df = _utils.ffill_zero(df)
    assert not (filled_df.iloc[5:10] == 0).any().any(), "Forward fill zero failed"


def test_standard_deviation():
    # Test for the StandardDeviation class
    df = generate_mock_data()
    std = StandardDeviation(df)
    annualized_std = std.annualize()
    assert isinstance(annualized_std, StandardDeviation), "Annualization of StandardDeviation failed"


def test_variance():
    # Test for the Variance class
    df = generate_mock_data()
    std = StandardDeviation(df)
    variance = std.to_variance()
    assert isinstance(variance, Variance), "Conversion to Variance failed"


def test_long_term_variance():
    # Test for calculating long-term variance
    df = generate_mock_data()
    variance = Variance(df)
    annualized_variance = variance.annualize()
    assert isinstance(annualized_variance, Variance), "Annualization of Variance failed"


def run_tests():
    print("Running tests...")
    test_ffill_zero()
    print("test_ffill_zero passed")

    test_standard_deviation()
    print("test_standard_deviation passed")

    test_variance()
    print("test_variance passed")

    test_long_term_variance()
    print("test_long_term_variance passed")

    print("All tests passed.")


if __name__ == "__main__":
    run_tests()