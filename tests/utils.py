import numpy as np
import types

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

def MockNestedFunction(
        outer : callable,
        innerName : str,
        **freeVars) -> callable:
    """
    Mock the inner functions of a given function with the provided free variables.

    Parameters:
    - outer (callable): The outer function whose inner functions are to be mocked.
    - innerName (str): The name of the inner function to be mocked.
    - freeVars (dict): Dictionary of free variables to be used in the inner function.

    Returns:
    - callable: The mocked function with the free variables.
    """

    def freeVar(val): # Yes there is something ironic about this
        def nested():
            return val
        return nested.__closure__[0]

    codeAttribute = '__code__'
    if isinstance(outer, (types.FunctionType, types.MethodType)):
        outer = outer.__getattribute__(codeAttribute)
    for const in outer.co_consts:
        if isinstance(const, types.CodeType) and const.co_name == innerName:
            return types.FunctionType(const, globals(), None, None, tuple(
                freeVar(freeVars[name]) for name in const.co_freevars))