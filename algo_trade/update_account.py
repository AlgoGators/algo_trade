import ipaddress
from algo_trade.ib_utils.account import Account
from decimal import Decimal
from algo_trade.ib_utils.api_handler import api_handler_context
from algo_trade.ib_utils.trading_algorithm import TradingAlgorithm
from algo_trade.ib_utils._enums import AdaptiveOrderPriority
from algo_trade.ib_utils._config import LOCALHOST, PORT, CLIENT_ID

def update_account(account : Account, order_priority : AdaptiveOrderPriority):
    with api_handler_context(ipaddress.ip_address(LOCALHOST), PORT, CLIENT_ID) as api_handler:
        api_handler.cancel_outstanding_orders()

        held_account = Account(api_handler.current_positions)

        api_handler.initialize_desired_contracts(account, 5)

        trades = account - held_account

        api_handler.place_orders(
            trades=trades,
            trading_algorithm=TradingAlgorithm(order_priority).adaptive_market_order
        )
