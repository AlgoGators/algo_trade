import numpy as np

def PriceSeries(
        volatility : float,
        expected_return : float,
        initial_price : float,
        n : int,
        seed : int = None) -> np.ndarray:
    """
    Generate a stochastic price series with a given volatility and drift.

    Parameters:
    - volatility (float): The volatility of the price series.
    - expected_return (float): The expected return (drift) of the price series.
    - initial_price (float): The initial price of the series.
    - n (int): Number of price points to generate.
    - seed (int, optional): Seed for the random number generator for reproducibility.

    Returns:
    - np.ndarray: Array containing the generated price series.
    """

    prices = np.zeros(n)
    prices[0] = initial_price

    # Initialize the random number generator with the given seed
    rng = np.random.default_rng(seed)

    for i in range(1, n):
        # Generate a random number from a standard normal distribution
        epsilon = rng.normal(0, 1)
        # Update the price based on the previous price, expected return, and volatility
        prices[i] = prices[i-1] * (1 + expected_return + volatility * epsilon)

    return prices