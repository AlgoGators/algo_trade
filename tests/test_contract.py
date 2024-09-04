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
from algo_trade.contract import Contract, Agg, ContractType, RollType, CATALOG, DATASET
import pytest
import databento as dbn
import toml
from dotenv import load_dotenv
import os

# Load the environment variables
load_dotenv()
if os.getenv('DATABENTO_API_KEY') is None:
    raise ValueError('DATABENTO_API_KEY not found in environment variables.')
else:
    print(f"API Key: {os.getenv('DATABENTO_API_KEY')}")

# Load the configuration file
config = toml.load('config/config.toml')


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

    Test:
        - The expiratoin property series is the same length as the timestamp series.
        - Each element is in in order... drop duplicates and check that each timestamped expiry is greater than the previous.
    """
    assert contract.expiration.shape[0] == contract.timestamp.shape[0]
    assert contract.expiration.drop_duplicates().is_monotonic_increasing