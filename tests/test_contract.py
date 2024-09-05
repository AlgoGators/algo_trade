"""
Contract Tests:
    - Test the Contract class in the future module.

Functions:
    - test_contract_init: Test the initialization of a Contract object.
    - test_contract_props: Test the OHLCV properties of a Contract object.
    - test_backadjusted: Test the backadjusted method within the Contract class.
    - test_contract_expiration: Test the expiration property

Author: Cole Rottenberg
Organization: AlgoGators Investment Fund
"""
import pandas as pd
import numpy as np
import pytest
import databento as dbn
from dotenv import load_dotenv
import os
from algo_trade.contract import Contract, Agg, ContractType, RollType, CATALOG, DATASET

# Load the environment variables
load_dotenv()
if os.getenv('DATABENTO_API_KEY') is None:
    raise ValueError('DATABENTO_API_KEY not found in environment variables.')
else:
    print(f"API Key: {os.getenv('DATABENTO_API_KEY')}")


@pytest.fixture
def contract() -> Contract:
    """
    Initialize a Contract object for testing.
    """
    contract: Contract = Contract('ES', dataset=DATASET.CME, schema=Agg.DAILY, catalog=CATALOG.DATABENTO)
    client: dbn.Historical = dbn.Historical()
    contract.construct(client=client, roll_type=RollType.CALENDAR, contract_type=ContractType.FRONT)
    return contract


# Test the initialization of a Contract object
def test_contract_init(contract: Contract):
    assert contract.instrument == 'ES'
    assert contract.dataset == DATASET.CME
    assert contract.schema == Agg.DAILY
    assert contract.catalog == CATALOG.DATABENTO


# Test the OHLCV properties of a Contract object
def test_contract_props(contract: Contract):
    assert contract.open is not None
    assert contract.high is not None
    assert contract.low is not None
    assert contract.close is not None
    assert contract.volume is not None
    # Check for increasing dates
    assert contract.timestamp.is_monotonic_increasing
    assert contract.timestamp.is_unique

# Test the backadjusted method within the Contract class
def test_backadjusted(contract: Contract):
    """
    Test the backadjusted method within the Contract class.
    """
    # TODO: Implement backadjusted as a property and turn the function into a ~pure~ function
    backadjusted = contract.backadjusted
    assert backadjusted is not None

# Test the expiration property
def test_contract_expiration(contract: Contract):
    """
    Test the expiration property.

    The _set_exp method is called in the construct method of the Contract class. Our test will revolve around this method and testing its functionality.

    We will need a dataframe with atleast an index of dates and a column of instrument ids. We then will need a dataframe with matching instrument ids and an expiration date with a non-daily frequency. We will then test to see if the returned expiration dataframe contains daily expiration dates.

    Test:
        - The expiratoin property series is the same length as the timestamp series.
        - Each element is in in order... drop duplicates and check that each timestamped expiry is greater than the previous.
    """

    dates = pd.date_range(start='2021-01-01', periods=5, freq='D')
    instrument_ids = [1, 1, 3, 3, 5]
    data = pd.DataFrame({"instrument_id": instrument_ids}, index=dates)

    instrument_ids = [1, 3, 5]
    expirations = pd.date_range(start='2021-01-31', periods=3, freq='M')
    dates = pd.date_range(start='2021-01-01', periods=3, freq='2D')

    definition = pd.DataFrame({"instrument_id": instrument_ids, "expiration": expirations}, index=dates)

    exp: pd.Series = contract._set_exp(data, definition)

    # Test 1: The expiration series index is the same is the data index
    assert len(exp) == len(data)

    # Test 2: The index of the expiration series is the same as the data index 
    assert exp.index.equals(data.index)

    # Test 3: The expiration series is in order
    assert exp.index.is_monotonic_increasing
