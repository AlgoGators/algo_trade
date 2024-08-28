import time
from ..src._contract import Contract
from ibapi.order import Order
from ..src._tag_value import TagValue
from ..src._enums import OrderType, AlgoStrategy, AdaptiveOrderPriority
from ..src.api_handler import IBAPI
from ..src._config import TIMEOUT

class AvailableAlgoParams:
    @staticmethod
    def FillAdaptiveParams(baseOrder : Order, priority : str):
        baseOrder.algoStrategy = AlgoStrategy.ADAPTIVE
        baseOrder.algoParams = []
        baseOrder.algoParams.append(TagValue("adaptivePriority", priority))

class TradingAlgorithm:
    def __init__(self, adaptive_priority : AdaptiveOrderPriority = None) -> None:
        self.adaptive_priority = adaptive_priority

    def getNewOrderId(self, lastOrderId : int, app : IBAPI) -> None:
        with app.condition:
            app.reqIds(-1)
            timed_out = not app.condition.wait(TIMEOUT)
        if timed_out:
            raise TimeoutError("Requesting new order ID timed out")

    def market_order(
            self,
            app : IBAPI,
            contract : Contract,
            order : Order) -> None:
        order.orderType = OrderType.MARKET
        self.getNewOrderId(app.nextValidOrderId, app)
        app.placeOrder(app.nextValidOrderId, contract, order)

    def adaptive_market_order(
            self,
            app : IBAPI,
            contract : Contract,
            order : Order) -> None:
        order.orderType = OrderType.MARKET
        self.getNewOrderId(app.nextValidOrderId, app)
        AvailableAlgoParams.FillAdaptiveParams(order, self.adaptive_priority)
        app.placeOrder(app.nextValidOrderId, contract, order)
