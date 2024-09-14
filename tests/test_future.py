"""
Testing Futures Instrument

Tests:
    - Test the initialization of a Contract object.
    - Test the OHLCV properties of a Contract object.
    - Test the backadjusted method within the Contract class.
    - Test the expiration property

Author: Cole Rottenberg
Organization: AlgoGators Investment Fund
"""

from algo_trade.instrument import Future
from algo_trade.contract import Agg, RollType, ContractType, CATALOG

import pytest
import pandas as pd
from dotenv import load_dotenv
import os

# Load the environment variables
load_dotenv()
if os.getenv('DATABENTO_API_KEY') is None:
    raise ValueError('DATABENTO_API_KEY not found in environment variables.')

# Future Fixture
@pytest.fixture
def future() -> Future:
    """
    Initialize a Future object for testing.
    """
    future: Future = Future("ES", "CME", multiplier=5.0)
    future.add_data(schema=Agg.DAILY, roll_type=RollType.CALENDAR, contract_type=ContractType.FRONT)
    return future

def test_future_init(future: Future):
    assert future.symbol == "ES"
    assert future.dataset == "CME"
    assert future.multiplier == 5.0
    assert future.contracts != {}
    assert future.front is not None

def test_front_props(future: Future):
    assert future.front.open is not None
    assert future.front.high is not None
    assert future.front.low is not None
    assert future.front.close is not None
    assert future.front.volume is not None
    # Check for increasing dates
    assert future.front.timestamp.is_monotonic_increasing
    assert future.front.timestamp.is_unique