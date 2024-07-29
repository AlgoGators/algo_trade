import sys

# Add parent directory to path
sys.path.append('../ib_utils')

import unittest
from unittest.mock import patch, MagicMock

import ipaddress
from src.api_handler import APIHandler
from src._contract import Contract
from ibapi.order import Order
from decimal import Decimal
from src._type_hints import ContractDetails
from src._enums import AccountSummaryTag
from unittesting._utils import position_is_equal, contract_is_equal

class TestAPIHandler(unittest.TestCase):
    def setUp(self) -> None:
        self.IP_ADDRESS = ipaddress.ip_address('127.0.0.1')
        self.PORT = 4002
        self.CLIENT_ID = 0
    
    @patch('src.api_handler.IBAPI')
    def test_connect(self, mock_IBAPI):
        api_handler = APIHandler(IP_ADDRESS=self.IP_ADDRESS, PORT=self.PORT, CLIENT_ID=self.CLIENT_ID)
        api_handler.connect()
        
        mock_IBAPI.return_value.connect.assert_called_once_with(str(self.IP_ADDRESS), self.PORT, self.CLIENT_ID)

    @patch('src.api_handler.IBAPI')
    def test_disconnect(self, mock_IBAPI):
        # Create an instance of APIHandler
        api_handler = APIHandler(self.IP_ADDRESS, self.PORT, self.CLIENT_ID)

        # Call the disconnect method
        api_handler.disconnect()

        # Assert that the disconnect method of IBAPI is called
        mock_IBAPI.return_value.disconnect.assert_called_once()
    
    @patch('src.api_handler.IBAPI')
    def test_get_current_positions(self, mock_IBAPI):
        # Create an instance of APIHandler
        api_handler = APIHandler(self.IP_ADDRESS, self.PORT, self.CLIENT_ID)

        positions_data = {1: [Contract(symbol='AAPL'), Decimal('100')],
                          2: [Contract(symbol='GOOG'), Decimal('200')]}

        # Mock the positions retrieval in IBAPI
        mock_IBAPI.return_value.positions = positions_data        

        def mock_req_positions():
            # Mock reqPositions method to populate positions after it's called
            mock_IBAPI.return_value.positions = positions_data

        mock_IBAPI.return_value.reqPositions.side_effect = mock_req_positions

        # Mock the get_contract_details method to return a contract
        mock_contract_details = {1: Contract(symbol='AAPL'), 2: Contract(symbol='GOOG')}
        
        contract_details_object_A = ContractDetails()
        contract_details_object_A.contract, contract_details_object_A.contract.conId = Contract(symbol='AAPL'), 1
        mock_contract_details : dict[str, ContractDetails] = {1: contract_details_object_A}
        contract_details_object_B = ContractDetails()
        contract_details_object_B.contract, contract_details_object_B.contract.conId = Contract(symbol='GOOG'), 2
        mock_contract_details[2] = contract_details_object_B
        
        def mock_get_contract_details(contract_id, contract):
            mock_IBAPI.return_value.contract_details = mock_contract_details

        mock_IBAPI.return_value.reqContractDetails.side_effect = mock_get_contract_details

        # Call the get_current_positions method
        result = api_handler.get_current_positions()

        # Assert that the positions are retrieved and processed correctly
        self.assertTrue(position_is_equal(result, positions_data))

        # Assert that the reqPositions method of IBAPI is called
        mock_IBAPI.return_value.reqPositions.assert_called_once()

    @patch('src.api_handler.IBAPI')
    def test_get_contract_details(self, mock_IBAPI):
        # Create an instance of APIHandler
        api_handler = APIHandler(self.IP_ADDRESS, self.PORT, self.CLIENT_ID)

        contractA = Contract(symbol='AAPL')
        contract_details_object = ContractDetails()
        contract_details_object.contract, contract_details_object.contract.conId = contractA, 1
        contract_details : dict[str, ContractDetails] = {1: contract_details_object}

        def mock_req_contract_details(reqId, contract):
            # Mock reqContractDetails method to populate contracts after it's called
            mock_IBAPI.return_value.contract_details = contract_details

        # Set side_effect of reqContractDetails to mock_req_contract_details function
        mock_IBAPI.return_value.reqContractDetails.side_effect = mock_req_contract_details

        # Call the get_contract_details method
        result = api_handler.get_contract_details(contractA)

        # Assert that the contract details are retrieved and processed correctly
        self.assertTrue(contract_is_equal(result[1], contract_details[1].contract))

        # Assert that the reqContractDetails method of IBAPI is called
        mock_IBAPI.return_value.reqContractDetails.assert_called_with(0, contractA)

    @patch('src.api_handler.IBAPI')
    def test_cancel_outstanding_orders(self, mock_IBAPI):
        # Create an instance of APIHandler
        api_handler = APIHandler(self.IP_ADDRESS, self.PORT, self.CLIENT_ID)

        # Call the cancel_outstanding_orders method
        api_handler.cancel_outstanding_orders()

        # Assert that the reqGlobalCancel method of IBAPI is called
        mock_IBAPI.return_value.reqGlobalCancel.assert_called_once()

    @patch('src.api_handler.IBAPI')
    def test_place_orders(self, mock_IBAPI):
        # Create an instance of APIHandler
        api_handler = APIHandler(self.IP_ADDRESS, self.PORT, self.CLIENT_ID)

        # Define mock trades
        mock_trades = [
            (Contract(), Order()),  # Trade 1
            (Contract(), Order()),  # Trade 2
        ]

        # Mock the cancel_outstanding_orders method
        api_handler.cancel_outstanding_orders = MagicMock()

        trading_algorithm_mock = MagicMock()
        trading_algorithm_mock.__name__ = "mockOrder"

        # Call the place_orders method
        api_handler.place_orders(mock_trades, trading_algorithm_mock)

        # Assert that cancel_outstanding_orders is called
        api_handler.cancel_outstanding_orders.assert_called_once()

        # Assert that the trading algorithm is called for each trade
        for contract, order in mock_trades:
            trading_algorithm_mock.assert_any_call(mock_IBAPI.return_value, contract, order)

    @patch('src.api_handler.IBAPI')
    def test_get_account_summary(self, mock_IBAPI):
        # Mocking
        mock_condition = MagicMock()
        mock_app = mock_IBAPI.return_value
        mock_app.condition = mock_condition
        mock_app.reqAccountSummary.return_value = None

        # Setup
        api_handler = APIHandler(IP_ADDRESS='127.0.0.1', PORT=1234, CLIENT_ID=1)

        # Calling the function under test
        result = api_handler._APIHandler__get_account_summary("TAG")

        # Assertions
        mock_app.reqAccountSummary.assert_called_once_with(0, "All", "TAG")
        mock_condition.wait.assert_called_once()
        self.assertEqual(result, mock_app.account_summary)

    @patch('src.api_handler.APIHandler._APIHandler__get_account_summary')
    def test_get_initial_margin(self, mock_get_account_summary):
        # Mocking
        mock_account_summary = {
            AccountSummaryTag.FULL_INIT_MARGIN_REQ: [('100.00', 'USD')]
        }
        mock_get_account_summary.return_value = mock_account_summary

        # Setup
        api_handler = APIHandler(IP_ADDRESS='127.0.0.1', PORT=1234, CLIENT_ID=1)

        # Calling the function under test
        result = api_handler.get_initial_margin()

        # Assertions
        self.assertEqual(result, {'USD': Decimal('100.00')})

    @patch('src.api_handler.APIHandler._APIHandler__get_account_summary')
    def test_get_maintenance_margin(self, mock_get_account_summary):
        # Mocking
        mock_account_summary = {
            AccountSummaryTag.FULL_MAINT_MARGIN_REQ: [('200.00', 'USD')]
        }
        mock_get_account_summary.return_value = mock_account_summary

        # Setup
        api_handler = APIHandler(IP_ADDRESS='127.0.0.1', PORT=1234, CLIENT_ID=1)

        # Calling the function under test
        result = api_handler.get_maintenance_margin()

        # Assertions
        self.assertEqual(result, {'USD': Decimal('200.00')})

    @patch('src.api_handler.APIHandler._APIHandler__get_account_summary')
    def test_get_cash_balances(self, mock_get_account_summary):
        # Mocking
        mock_account_summary = {
            "TotalCashBalance" : [('1000.00', 'USD'), ('2000.00', 'EUR')]
        }
        mock_get_account_summary.return_value = mock_account_summary

        # Setup
        api_handler = APIHandler(IP_ADDRESS='127.0.0.1', PORT=1234, CLIENT_ID=1)

        # Calling the function under test
        result = api_handler.get_cash_balances()

        # Assertions
        self.assertEqual(result, {'USD': Decimal('1000.00'), 'EUR': Decimal('2000.00')})

    @patch('src.api_handler.APIHandler._APIHandler__get_account_summary')
    def test_get_exchange_rates(self, mock_get_account_summary):
        # Mocking
        mock_account_summary = {
            "ExchangeRate": [('1.1000', 'EUR'), ('0.9000', 'USD')]
        }
        mock_get_account_summary.return_value = mock_account_summary

        # Setup
        api_handler = APIHandler(IP_ADDRESS='127.0.0.1', PORT=1234, CLIENT_ID=1)

        # Calling the function under test
        result = api_handler.get_exchange_rates()

        # Assertions
        self.assertEqual(result, {'EUR': Decimal('1.1000'), 'USD': Decimal('0.9000')})

if __name__ == '__main__':
    unittest.main(failfast=True)