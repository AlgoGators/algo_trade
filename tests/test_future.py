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
import asyncio
import pytest
import os

from dotenv import load_dotenv
import pandas as pd # type: ignore

from algo_trade.contract import Agg, RollType, ContractType, CATALOG, DATASET
from algo_trade.instrument import Future


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
    future: Future = Future("ES", DATASET.GLOBEX, multiplier=5.0, exchange=DATASET.GLOBEX, currency="USD")

    # for future in futures:
    #     future.contracts = CATALOG[future.symbol]
    #     future.front =

    asyncio.run(
        future.add_data_async(
            schema=Agg.DAILY,
            roll_type=RollType.CALENDAR,
            contract_type=ContractType.FRONT
        )
    )

    return future

def test_future_init(future: Future) -> None:
    assert future.symbol == "ES"
    assert future.dataset == DATASET.GLOBEX
    assert future.multiplier == 5.0
    assert future.contracts != {}
    assert future.front is not None

def test_front_props(future: Future) -> None:
    assert future.front.open is not None
    assert future.front.high is not None
    assert future.front.low is not None
    assert future.front.close is not None
    assert future.front.volume is not None
    # Check for increasing dates
    assert future.front.timestamp.is_monotonic_increasing
    assert future.front.timestamp.is_unique