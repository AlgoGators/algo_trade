import ipaddress

from decimal import Decimal

from algo_trade.ib_utils.account import Account, Position, Contract
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

if __name__ == "__main__":
    new_positions = [
        Position(Contract(symbol='MES', multiplier='5', exchange='CME', currency='USD', secType='FUT'), Decimal(2)),
        Position(Contract(symbol='MYM', multiplier='0.5', exchange='CBOT', currency='USD', secType='FUT'), Decimal(-2))
    ]
    new_account = Account(new_positions)

    update_account(account=new_account, order_priority=AdaptiveOrderPriority.NORMAL)
