"""
Test the the contract with Norgate Data.
"""

import pytest
import pandas as pd
from algo_trade.contract import Contract, Agg, ContractType, RollType, DATASET, CATALOG


@pytest.fixture
def contract() -> Contract:
    """
    Returns a contract object constructed with Norgate Data.

    Returns:
        Contract: A contract object.
    """
    contract: Contract = Contract(
        instrument="ES", dataset=DATASET.GLOBEX, schema=Agg.DAILY, catalog=CATALOG.NORGATE
    )
    contract.construct_norgate()
    return contract


def test_contract_norgate(contract: Contract):
    """
    Test the contract with Norgate Data.

    Args:
        contract (Contract): A contract object.
    """

    # Check for proper initialization
    assert contract.instrument == "ES"
    assert contract.dataset == DATASET.GLOBEX
    assert contract.schema == Agg.DAILY
    assert contract.catalog == CATALOG.NORGATE

    # Check for proper OHLCV properties
    assert contract.open.equals(contract.data["Open"])
    assert contract.high.equals(contract.data["High"])
    assert contract.low.equals(contract.data["Low"])
    assert contract.close.equals(contract.data["Unadj_Close"])
    assert contract.volume.equals(contract.data["Volume"])
    assert contract.open_interest.equals(contract.data["Open Interest"])

    # Check for backadjusted prices being equal to the close prices
    assert contract.backadjusted.equals(contract.data["Close"])

    # Check for proper timestamp and expiration
    # TODO: Implement a check for the timestamp and expiration... having issues with the timestamp being an index and not a series
    # assert contract.timestamp.equals(pd.to_datetime(contract.data["Date"] + " " + contract.data["Time"]))
    assert contract.timestamp.is_monotonic_increasing
    assert contract.timestamp.is_unique
    assert contract.expiration.equals(
        pd.to_datetime(contract.data["Delivery Month"], format="%Y%m")
    )
