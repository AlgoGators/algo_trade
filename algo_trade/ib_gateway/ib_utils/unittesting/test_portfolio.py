import sys

# Add parent directory to path
sys.path.append('../ib_utils')

import unittest
from unittest.mock import MagicMock, patch, Mock
from datetime import datetime, date, timedelta
from src._contract import Contract
from ibapi.order import Order
from src.api_handler import APIHandler
from src.portfolio import Portfolio
from decimal import Decimal
from src._enums import OrderAction
from unittesting._utils import trades_is_equal

class TestPortfolio(unittest.TestCase):
    def test_get_expiring_contracts(self):
        # Mock the APIHandler
        api_handler_mock = MagicMock(spec=APIHandler)

        # Create a Portfolio instance
        portfolio = Portfolio(api_handler_mock)

        # mock contracts with expiration dates
        contract1 = Contract(lastTradeDateOrContractMonth=date.today()+timedelta(days=5))
        contract2 = Contract(lastTradeDateOrContractMonth=date.today()+timedelta(days=10))
        contract3 = Contract(lastTradeDateOrContractMonth=date.today()+timedelta(days=15))

        # Create a dictionary of mock contracts
        contracts = {
            "Contract1": contract1,
            "Contract2": contract2,
            "Contract3": contract3
        }

        # Set the expected result
        expected_expiring_contracts = {
            "Contract1": contract1,
            "Contract2": contract2
        }

        # Mock the current date
        today = date.today()

        # Set up the mock return value for today's date
        today_mock = MagicMock(spec=date)
        today_mock.today.return_value = today

        # Patch the date.today() function to return the mock date
        with unittest.mock.patch('src.portfolio.datetime.date', today_mock):
            # Call the get_expiring_contracts method
            expiring_contracts = portfolio.get_expiring_contracts(contracts, min_DTE=11)

            # Assert that the result matches the expected expiring contracts
            self.assertEqual(expiring_contracts, expected_expiring_contracts)

    def test_get_expirations(self): #! need to make a little more comprehensive
        # Mock the APIHandler
        api_handler_mock = MagicMock(spec=APIHandler)

        # Create a Portfolio instance
        portfolio = Portfolio(api_handler_mock)

        # Create a mock Contract object for testing
        mock_contract = Contract()
        mock_contract.symbol = "Contract1"  # Use an existing key in mock_contract_details

        # Define three expiration dates with actual date values
        today = datetime.now().date()
        expiration1 = (today + timedelta(days=10)).strftime("%Y%m%d")
        expiration2 = (today + timedelta(days=20)).strftime("%Y%m%d")
        expiration3 = (today + timedelta(days=30)).strftime("%Y%m%d")

        # Mock the return value of get_contract_details method
        mock_contract_details = {
            "Contract1": Contract(lastTradeDateOrContractMonth=expiration1),
            "Contract2": Contract(lastTradeDateOrContractMonth=expiration2),
            "Contract3": Contract(lastTradeDateOrContractMonth=expiration3)
        }
        api_handler_mock.get_contract_details.return_value = mock_contract_details

        # Call the get_expirations method
        expirations = portfolio.get_expirations(mock_contract)

        # Assert that the result matches the expected expiration dates
        expected_expirations = [expiration1, expiration2, expiration3]
        self.assertEqual(expirations, expected_expirations)

    def test_get_desired_positions_with_valid_expirations(self):
        # Create a mock contract
        mock_contract = Contract(symbol="MES")

        # Define the expiration dates
        today = datetime.today().date()
        expirations = [(today + timedelta(days=30)).strftime("%Y%m%d"), 
                   (today + timedelta(days=60)).strftime("%Y%m%d"), 
                   (today + timedelta(days=90)).strftime("%Y%m%d")]

        # Mock the get_contract_details method of APIHandler
        with patch.object(APIHandler, 'get_contract_details') as mock_get_contract_details:
            mock_get_contract_details.return_value = {1: mock_contract}

            # Mock the get_expirations method of Portfolio
            with patch.object(Portfolio, 'get_expirations') as mock_get_expirations:
                mock_get_expirations.return_value = expirations

                # Mock the APIHandler constructor
                api_handler_mock = Mock()
                api_handler_mock.get_contract_details.return_value = {1: mock_contract}
                
                # Initialize the portfolio
                portfolio = Portfolio(api_handler_mock)

                # Call the function under test
                desired_positions = portfolio.get_desired_positions({mock_contract: Decimal('100')}, min_DTE=15)

                # Assert that the desired positions are correct
                self.assertEqual(len(desired_positions), 1)
                contract_id, desired_position = desired_positions.popitem()
                self.assertEqual(desired_position[0], mock_contract)
                self.assertEqual(desired_position[1], Decimal('100'))  # Check the position quantity

    def test_get_required_trades(self):
        # Mock held positions
        held_positions = {
            'VGT': [Contract(symbol='VGT'), Decimal('20')],
            'AAPL': [Contract(symbol='AAPL'), Decimal('100')],
            'GOOG': [Contract(symbol='GOOG'), Decimal('200')],
            'SPY': [Contract(symbol='SPY'), Decimal('-100')]
        }
        
        # Mock desired positions
        desired_positions = {
            'VGT': [Contract(symbol='VGT'), Decimal('20')],
            'AAPL': [Contract(symbol='AAPL'), Decimal('150')],
            'GOOG': [Contract(symbol='GOOG'), Decimal('180')],
            'MSFT': [Contract(symbol='MSFT'), Decimal('300')]
        }
        
        # Create a MagicMock for the APIHandler
        api_handler_mock = MagicMock()
        
        # Create a Portfolio instance
        portfolio = Portfolio(api_handler_mock)
        
        # Call the get_required_trades method
        trades = portfolio.get_required_trades(held_positions, desired_positions)
        
        # Create MagicMock objects for the expected trades
        expected_trades = [
            (Contract(symbol='AAPL'), Order()),
            (Contract(symbol='GOOG'), Order()),
            (Contract(symbol='MSFT'), Order()),
            (Contract(symbol='SPY'), Order())
        ]
        
        # Set the attributes of MagicMock objects to match the expected values
        expected_trades[0][1].action = OrderAction.BUY
        expected_trades[0][1].totalQuantity = Decimal('50')
        expected_trades[1][1].action = OrderAction.SELL
        expected_trades[1][1].totalQuantity = Decimal('20')
        expected_trades[2][1].action = OrderAction.BUY
        expected_trades[2][1].totalQuantity = Decimal('300')
        expected_trades[3][1].action = OrderAction.BUY
        expected_trades[3][1].totalQuantity = Decimal('100')

        self.assertTrue(trades_is_equal(trades, expected_trades))
    
    def test_get_required_trades_no_held_positions(self):
        # Mock held positions
        held_positions = {}
        
        # Mock desired positions
        desired_positions = {
            'VGT': [Contract(symbol='VGT'), Decimal('20')],
            'AAPL': [Contract(symbol='AAPL'), Decimal('150')],
            'GOOG': [Contract(symbol='GOOG'), Decimal('180')],
            'MSFT': [Contract(symbol='MSFT'), Decimal('300')]
        }
        
        # Create a MagicMock for the APIHandler
        api_handler_mock = MagicMock()
        
        # Create a Portfolio instance
        portfolio = Portfolio(api_handler_mock)
        
        # Call the get_required_trades method
        trades = portfolio.get_required_trades(held_positions, desired_positions)
        
        # Create MagicMock objects for the expected trades
        expected_trades = [
            (Contract(symbol='AAPL'), Order()),
            (Contract(symbol='GOOG'), Order()),
            (Contract(symbol='MSFT'), Order()),
            (Contract(symbol='VGT'), Order())
        ]
        
        # Set the attributes of MagicMock objects to match the expected values
        expected_trades[0][1].action = OrderAction.BUY
        expected_trades[0][1].totalQuantity = Decimal('150')
        expected_trades[1][1].action = OrderAction.BUY
        expected_trades[1][1].totalQuantity = Decimal('180')
        expected_trades[2][1].action = OrderAction.BUY
        expected_trades[2][1].totalQuantity = Decimal('300')
        expected_trades[3][1].action = OrderAction.BUY
        expected_trades[3][1].totalQuantity = Decimal('20')

        self.assertTrue(trades_is_equal(trades, expected_trades))

    def test_get_required_trades_no_desired_positions(self):
        # Mock held positions
        held_positions = {
            'VGT': [Contract(symbol='VGT'), Decimal('20')],
            'AAPL': [Contract(symbol='AAPL'), Decimal('150')],
            'GOOG': [Contract(symbol='GOOG'), Decimal('180')],
            'MSFT': [Contract(symbol='MSFT'), Decimal('300')]
        }
        
        # Mock desired positions
        desired_positions = {}
        
        # Create a MagicMock for the APIHandler
        api_handler_mock = MagicMock()
        
        # Create a Portfolio instance
        portfolio = Portfolio(api_handler_mock)
        
        # Call the get_required_trades method
        trades = portfolio.get_required_trades(held_positions, desired_positions)
        
        # Create MagicMock objects for the expected trades
        expected_trades = [
            (Contract(symbol='VGT'), Order()),
            (Contract(symbol='AAPL'), Order()),
            (Contract(symbol='GOOG'), Order()),
            (Contract(symbol='MSFT'), Order())
        ]
        
        # Set the attributes of MagicMock objects to match the expected values
        expected_trades[0][1].action = OrderAction.SELL
        expected_trades[0][1].totalQuantity = Decimal('20')
        expected_trades[1][1].action = OrderAction.SELL
        expected_trades[1][1].totalQuantity = Decimal('150')
        expected_trades[2][1].action = OrderAction.SELL
        expected_trades[2][1].totalQuantity = Decimal('180')
        expected_trades[3][1].action = OrderAction.SELL
        expected_trades[3][1].totalQuantity = Decimal('300')

        self.assertTrue(trades_is_equal(trades, expected_trades))

    def test_get_required_trades_no_position(self):
        # Mock held positions
        held_positions = {}
        
        # Mock desired positions
        desired_positions = {}
        
        # Create a MagicMock for the APIHandler
        api_handler_mock = MagicMock()
        
        # Create a Portfolio instance
        portfolio = Portfolio(api_handler_mock)
        
        # Call the get_required_trades method
        trades = portfolio.get_required_trades(held_positions, desired_positions)
        
        # Create MagicMock objects for the expected trades
        expected_trades = []

        self.assertTrue(trades_is_equal(trades, expected_trades))



if __name__ == "__main__":
    unittest.main(failfast=True)
