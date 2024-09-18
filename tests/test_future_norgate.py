import pytest
from algo_trade.instrument import Instrument, Future
from algo_trade.contract import DATASET

@pytest.fixture
def future():
    future: Future = Future("ES", DATASET.GLOBEX, multiplier=50, currency="USD", exchange="CME")
    future.add_norgate_data("Norgate")
    return future


def test_future_norgate(future):
    """
    Test the basic functionality of the future instrument

    Args:
      - future: Future - The future instrument
      
    Returns:
      - None
    """
    assert future.symbol == "ES"
    assert future.dataset == DATASET.GLOBEX 
    assert future.multiplier == 50

    # Test the properties of the future
    assert future.front is not None
    assert future.contracts is not None
    assert future.price is not None