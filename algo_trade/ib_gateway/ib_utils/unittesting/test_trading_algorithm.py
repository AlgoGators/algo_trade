import sys

# Add parent directory to path
sys.path.append('../ib_utils')

import unittest
from unittest.mock import MagicMock
from src._contract import Contract
from ibapi.order import Order
from src._enums import AdaptiveOrderPriority, AlgoStrategy, OrderType
from src.api_handler import IBAPI
from src.trading_algorithm import AvailableAlgoParams, TradingAlgorithm
from ibapi.tag_value import TagValue

class TestTradingAlgorithm(unittest.TestCase):
    def test_fill_adaptive_params(self):
        # Create a base order and set its attributes
        base_order = Order()
        base_order.algoStrategy = ""
        base_order.algoParams = []

        # Mock the priority
        priority = AdaptiveOrderPriority.NORMAL

        # Call the FillAdaptiveParams method
        AvailableAlgoParams.FillAdaptiveParams(base_order, priority)

        # Assert that the algoStrategy is set to Adaptive
        self.assertEqual(base_order.algoStrategy, AlgoStrategy.ADAPTIVE)

        # Assert that the algoParams list contains the correct TagValue
        self.assertEqual(len(base_order.algoParams), 1)
        self.assertEqual(base_order.algoParams[0].tag, "adaptivePriority")
        self.assertEqual(base_order.algoParams[0].value, priority.value)

    def test_market_order(self):
        # Create mock objects
        app_mock = MagicMock(spec=IBAPI)
        app_mock.condition = MagicMock()
        app_mock.nextValidOrderId = 12345  # Add nextValidOrderId attribute
        contract_mock = MagicMock(spec=Contract)
        order_mock = MagicMock(spec=Order)
        
        # Create a TradingAlgorithm instance
        trading_algo = TradingAlgorithm()

        def mock_reqIds(value):
            app_mock.nextValidOrderId += 1

        app_mock.reqIds.side_effect = mock_reqIds

        # Call the market_order method
        trading_algo.market_order(app_mock, contract_mock, order_mock)

        # Assert that reqIds and placeOrder methods are called
        app_mock.reqIds.assert_called_once_with(-1)
        app_mock.placeOrder.assert_called_once_with(app_mock.nextValidOrderId, contract_mock, order_mock)

        # Assert that orderType is set to MARKET
        self.assertEqual(order_mock.orderType, OrderType.MARKET)    

    def test_adaptive_market_order(self):
        # Create mock objects
        app_mock = MagicMock(spec=IBAPI)
        app_mock.condition = MagicMock()
        app_mock.nextValidOrderId = 12345  # Add nextValidOrderId attribute
        contract_mock = MagicMock(spec=Contract)
        order_mock = MagicMock(spec=Order)
        
        # Create a TradingAlgorithm instance with adaptive priority
        trading_algo = TradingAlgorithm(adaptive_priority=AdaptiveOrderPriority.NORMAL)

        def mock_reqIds(value):
            app_mock.nextValidOrderId += 1

        app_mock.reqIds.side_effect = mock_reqIds

        # Call the adaptive_market_order method
        trading_algo.adaptive_market_order(app_mock, contract_mock, order_mock)

        # Assert that reqIds and placeOrder methods are called
        app_mock.reqIds.assert_called_once_with(-1)
        app_mock.placeOrder.assert_called_once_with(app_mock.nextValidOrderId, contract_mock, order_mock)

        # Assert that orderType is set to MARKET
        self.assertEqual(order_mock.orderType, OrderType.MARKET)

        # Assert that algoStrategy and algoParams are set correctly
        self.assertEqual(order_mock.algoStrategy, AlgoStrategy.ADAPTIVE)
        expected_params = [TagValue("adaptivePriority", AdaptiveOrderPriority.NORMAL)]

        self.assertEqual(order_mock.algoParams, expected_params)

if __name__ == "__main__":
    unittest.main(failfast=True)
