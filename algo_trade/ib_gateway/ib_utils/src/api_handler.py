import ipaddress
import typing
from decimal import Decimal
import time
import threading
import logging
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.client import EClient
from ibapi.order_state import OrderState
from ibapi.wrapper import EWrapper
from ..src._contract import Contract
from ..src._config import TIMEOUT, WAIT_FOR_TRADES, WAIT_FOR_TRADES_TIMEOUT
from ..src._type_hints import ContractDetails
from ..src._enums import AccountSummaryTag

threading.TIMEOUT_MAX = TIMEOUT

class IBAPI(EClient, EWrapper):
    def __init__(self, condition : threading.Condition) -> None:
        EClient.__init__(self, self)
        self.positions : dict[str, list[Contract, Decimal]] = {}
        self.contract_details : dict[str, ContractDetails] = {}
        self.contracts_margin : dict[str, Decimal] = {}
        self.account_summary : dict[str, list[tuple[str|float|Decimal|int, str]]] = {}
        self.open_orders : dict[int, list[str, str, Decimal, str]] = {}
        self.condition = condition

    def contractDetails(self, reqId : int, contractDetails : ContractDetails) -> None:
        self.contract_details[contractDetails.contract.conId] = contractDetails

    def contractDetailsEnd(self, reqId: int): 
        with self.condition:
            self.condition.notify()

    def openOrder(self, orderId: int, contract: Contract, order: Order, orderState: OrderState):
        self.open_orders[orderId] = [contract.symbol, order.action, order.totalQuantity, orderState.status]

    def openOrderEnd(self):
        with self.condition:
            self.condition.notify()

    def position(self, account : str, contract : Contract, position : Decimal, avgCost : Decimal) -> None:
        self.positions[contract.conId] = [Contract(contract=contract), position] #* converts to our abstract of contract

    def positionEnd(self) -> None:
        with self.condition:
            self.condition.notify()

    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        if tag in self.account_summary:
            self.account_summary[tag].append((value, currency))
        else:
            self.account_summary[tag] = [(value, currency)]

    def accountSummaryEnd(self, reqId: int) -> None:
        with self.condition:
            self.condition.notify()

    def nextValidId(self, orderId: int) -> None:
        super().nextValidId(orderId)
        with self.condition:
            self.nextValidOrderId = orderId
            logging.info(f"nextValidOrderId: {orderId}")
            self.condition.notify()

def run_loop(app : IBAPI) -> None: app.run()

class APIHandler:
    def __init__(
            self,
            IP_ADDRESS : ipaddress.IPv4Address,
            PORT : int,
            CLIENT_ID : int) -> None:
        self.app : IBAPI = IBAPI(condition=threading.Condition())
        self.IP_ADDRESS : ipaddress.IPv4Address = IP_ADDRESS
        self.PORT : int = PORT
        self.CLIENT_ID : int = CLIENT_ID
        self.api_thread : threading.Thread

    def __await_connection(self) -> None:
        with self.app.condition:
            logging.warning("Waiting for connection to TWS")
            timed_out = not self.app.condition.wait()
        if timed_out or not self.app.isConnected():
            raise ConnectionError("Failed to connect to TWS")
        logging.warning("Connected to TWS")

    def connect(self) -> None:
        self.app.connect(str(self.IP_ADDRESS), self.PORT, self.CLIENT_ID)
        self.api_thread = threading.Thread(
            target=run_loop, args=(self.app,), daemon=True)
        self.api_thread.start()
        self.__await_connection()

    def disconnect(self) -> None: self.app.disconnect()

    def __get_account_summary(self, tag : AccountSummaryTag) -> dict[str, list[tuple[str|float|Decimal|int, str]]]:
        with self.app.condition:
            self.app.account_summary = {} # Clear account summary dictionary
            self.app.reqAccountSummary(0, "All", tag)
            timed_out = not self.app.condition.wait()
        if timed_out:
            raise TimeoutError(f"Failed to retrieve account summary for {tag}")
        return self.app.account_summary

    def get_initial_margin(self) -> tuple[str, Decimal]:
        initial_margin = self.__get_account_summary(AccountSummaryTag.FULL_INIT_MARGIN_REQ)[AccountSummaryTag.FULL_INIT_MARGIN_REQ][0]
        return {initial_margin[1] : Decimal(initial_margin[0])}

    def get_maintenance_margin(self) -> tuple[str, Decimal]:
        maintenance_margin = self.__get_account_summary(AccountSummaryTag.FULL_MAINT_MARGIN_REQ)[AccountSummaryTag.FULL_MAINT_MARGIN_REQ][0]
        return {maintenance_margin[1] : Decimal(maintenance_margin[0])}

    def get_cash_balances(self) -> dict[str, Decimal]:
        account_summary = self.__get_account_summary(AccountSummaryTag.LEDGER)
        return {currency: Decimal(value) for tag, values in account_summary.items() if tag == "TotalCashBalance" for value, currency in values}

    def get_exchange_rates(self) -> dict[str, Decimal]:
        account_summary = self.__get_account_summary(AccountSummaryTag.LEDGER)
        return {currency: Decimal(value) for tag, values in account_summary.items() if tag == "ExchangeRate" for value, currency in values}

    def get_current_positions(self) -> dict[str, list[Contract, Decimal]]:
        with self.app.condition:
            self.app.positions = {} # Clear positions dictionary
            self.app.reqPositions()
            timed_out = not self.app.condition.wait()
        if timed_out:
            raise TimeoutError("Failed to retrieve current positions")

        positions : dict[str, list[Contract, Decimal]] = self.app.positions

        #! necessary to get exchange and other pieces of data for the contract
        #* hate this, but contract generated doesn't exist in their own DB
        for conId, (_, quantity) in positions.items():
            contract_by_ID = Contract(conId=conId)
            positions[conId] = [self.get_contract_details(contract_by_ID)[conId], quantity]
        return positions

    def get_contract_details(
            self,
            contract : Contract) -> dict[str, Contract]:
        with self.app.condition:
            self.app.contract_details = {} # Clear contracts dictionary
            self.app.reqContractDetails(0, contract)
            timed_out = not self.app.condition.wait()
        if timed_out:
            raise TimeoutError("Failed to retrieve contract details")
        contracts : dict[str, Contract] = {conId : contractDetails.contract for conId, contractDetails in self.app.contract_details.items()}

        return contracts

    def __wait_for_trades(self) -> None:
        start_time = time.time()
        while time.time() - start_time < WAIT_FOR_TRADES_TIMEOUT:
            with self.app.condition:
                self.app.open_orders = {}
                self.app.reqOpenOrders()
                timed_out = not self.app.condition.wait()
            if timed_out:
                raise TimeoutError("Failed to retrieve open orders")
            if not self.app.open_orders:
                return
            orders = "\n\t\t\t\t  " + "\n\t\t\t\t  ".join([f"{status.upper()}: {action} {quantity} {symbol} | OrderID: #{orderId}" for orderId, (symbol, action, quantity, status) in self.app.open_orders.items()])
            logging.warning(f"Waiting for orders to fill: {orders}")
            time.sleep(1)
        self.cancel_outstanding_orders()
        raise TimeoutError("Trades took too long to fill")

    def cancel_outstanding_orders(self) -> None:
        self.app.reqGlobalCancel()

    def place_orders(
            self,
            trades : list[tuple[Contract, Order]],
            trading_algorithm : typing.Callable) -> None:
        self.cancel_outstanding_orders()
        for contract, order in trades:
            logging.warning(f"{order.action} {order.totalQuantity} {contract.symbol} {contract.lastTradeDateOrContractMonth} on {contract.exchange} using {trading_algorithm.__name__}")
            trading_algorithm(self.app, contract, order)
        if WAIT_FOR_TRADES:
            self.__wait_for_trades()
