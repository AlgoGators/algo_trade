import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

import pandas as pd # type: ignore

from algo_trade.contract import DATASET, Contract
from algo_trade.instrument import SecurityType, Instrument, Future

class TestSecurityType(unittest.TestCase):
    def test_security_type_enum(self) -> None:
        """Test if SecurityType enum can be properly instantiated and resolves correctly."""
        self.assertEqual(SecurityType.FUTURE.name, "FUTURE")
        self.assertEqual(SecurityType.FUTURE.string, "FUT")

    def test_security_type_obj(self) -> None:
        """Test if SecurityType resolves to the correct object class."""
        self.assertEqual(SecurityType.FUTURE.obj, Future)

    def test_security_type_from_str(self) -> None:
        """Test the from_str method for valid and invalid string input."""
        self.assertEqual(SecurityType.from_str("FUTURE"), SecurityType.FUTURE)
        with self.assertRaises(KeyError):
            SecurityType.from_str("INVALID_TYPE")


class TestInstrument(unittest.TestCase):
    def setUp(self) -> None:
        """Set up a mock Instrument instance for testing."""
        self.instrument = Instrument(
            symbol="AAPL",
            dataset=DATASET.GLOBEX,
            currency="USD",
            exchange="NASDAQ"
        )

    def test_instrument_initialization(self) -> None:
        """Test the initialization of an Instrument."""
        self.assertEqual(self.instrument.symbol, "AAPL")
        self.assertEqual(self.instrument.dataset, DATASET.GLOBEX)
        self.assertEqual(self.instrument.currency, "USD")
        self.assertEqual(self.instrument.exchange, "NASDAQ")

    def test_get_symbol(self) -> None:
        """Test get_symbol method."""
        self.assertEqual(self.instrument.get_symbol(), "AAPL")

    def test_get_collection(self) -> None:
        """Test get_collection method."""
        self.assertEqual(self.instrument.get_collection(), ("AAPL", DATASET.GLOBEX))

    def test_price_setter_getter(self) -> None:
        """Test price setter and getter with valid and invalid inputs."""
        prices = pd.Series([100, 105, 110])
        self.instrument.price = prices
        with self.assertRaises(NotImplementedError):
            _ = self.instrument.price  # Price getter is not implemented

        with self.assertRaises(ValueError):
            self.instrument.price = "Not a pd.Series"  # Invalid price input

    def test_not_implemented_percent_returns(self) -> None:
        """Test if percent_returns property raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            _ = self.instrument.percent_returns


class TestFuture(unittest.TestCase):
    def setUp(self) -> None:
        """Set up a mock Future instance for testing."""
        self.future = Future(
            symbol="CL",
            dataset=DATASET.GLOBEX,
            currency="USD",
            exchange="NYMEX",
            multiplier=1000
        )
    
    def test_future_initialization(self) -> None:
        """Test the initialization of a Future."""
        self.assertEqual(self.future.symbol, "CL")
        self.assertEqual(self.future.dataset, DATASET.GLOBEX)
        self.assertEqual(self.future.currency, "USD")
        self.assertEqual(self.future.exchange, "NYMEX")
        self.assertEqual(self.future.multiplier, 1000)

    def test_contract_setters_and_getters(self) -> None:
        """Test the front and back contract properties."""
        mock_contract = MagicMock(spec=Contract)
        self.future.front = mock_contract
        self.assertEqual(self.future.front, mock_contract)
        
        with self.assertRaises(ValueError):
            _ = self.future.back  # Back contract should raise ValueError when not set

        self.future.back = mock_contract
        self.assertEqual(self.future.back, mock_contract)

    def test_future_price_setter(self) -> None:
        """Test setting and getting prices for a Future."""
        prices = pd.Series([50, 55, 60])
        self.future.price = prices
        pd.testing.assert_series_equal(self.future.price, prices)

    def test_add_norgate_data(self) -> None:
        """Test add_norgate_data adds contract and sets front."""
        #@ TODO

    def test_add_data_async(self) -> None:
        """Test add_data_async adds contract and sets front asynchronously."""
        #@ TODO


if __name__ == '__main__':
    unittest.main()
