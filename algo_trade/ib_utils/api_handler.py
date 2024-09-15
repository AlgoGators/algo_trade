import typing
import time
import threading
import logging
import datetime
from decimal import Decimal
from contextlib import contextmanager
import ipaddress
from dataclasses import dataclass

from ibapi.client import EClient
from ibapi.order_state import OrderState
from ibapi.wrapper import EWrapper
from ibapi.order import Order

from algo_trade.ib_utils._enums import AccountSummaryTag
from algo_trade.ib_utils._contract import Contract
from algo_trade.ib_utils._config import TIMEOUT, WAIT_FOR_TRADES, WAIT_FOR_TRADES_TIMEOUT
from algo_trade.ib_utils._type_hints import ContractDetails
from algo_trade.ib_utils.account import Account, Position, Trade
from algo_trade.ib_utils.error_codes import ErrorCodes

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

TickerId = int

# @dataclass
# class ConnectionStatus():
#     MARKET_DATA_FARM : bool = False
#     HMDS_DATA_FARM : bool = False # Historical Market Data
#     SEC_DEF_DATA_FARM : bool = False

#     @property
#     def is_connected(self):
#         return self.MARKET_DATA_FARM & self.HMDS_DATA_FARM & self.SEC_DEF_DATA_FARM
    
class ConnectionStatus():
    def __init__(self, condition : threading.Condition) -> None:
        self.condition = condition
        self._HMDS_DATA_FARM : bool = False
        self._SEC_DEF_DATA_FARM : bool = False
        self._MARKET_DATA_FARM : bool = False
        self._is_connected : bool = False

    def update_connected_status(self) -> None:
        if self.HMDS_DATA_FARM & self.SEC_DEF_DATA_FARM & self.MARKET_DATA_FARM:
            with self.condition:
                self.condition.notify()
            self._is_connected = True

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def MARKET_DATA_FARM(self) -> bool:
        return self._MARKET_DATA_FARM
    
    @MARKET_DATA_FARM.setter
    def MARKET_DATA_FARM(self, value : bool) -> None:
        self._MARKET_DATA_FARM = value
        self.update_connected_status()
    
    @property
    def HMDS_DATA_FARM(self) -> bool:
        return self._HMDS_DATA_FARM
    
    @HMDS_DATA_FARM.setter
    def HMDS_DATA_FARM(self, value : bool) -> None:
        self._HMDS_DATA_FARM = value
        self.update_connected_status()

    @property
    def SEC_DEF_DATA_FARM(self) -> bool:
        return self._SEC_DEF_DATA_FARM
    
    @SEC_DEF_DATA_FARM.setter
    def SEC_DEF_DATA_FARM(self, value : bool) -> None:
        self._SEC_DEF_DATA_FARM = value
        self.update_connected_status()

class IBAPI(EClient, EWrapper):
    def __init__(self, condition : threading.Condition) -> None:
        EClient.__init__(self, self)
        self.positions : list[Position] = [] #dict[str, list[Contract, Decimal]] = {}
        self.contract_details : dict[str, ContractDetails] = {}
        self.contracts_margin : dict[str, Decimal] = {}
        self.account_summary : dict[str, list[tuple[str|float|Decimal|int, str]]] = {}
        self.open_orders : dict[int, list[str, str, Decimal, str]] = {}
        self.condition = condition
        self.connection_status : ConnectionStatus = ConnectionStatus(condition=condition)

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
        #* converts to our abstract of contract
        #* note ibkr means quantity when it says position :/
        self.positions.append(Position(Contract(contract=contract), position))

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

    def error(self, reqId : TickerId, errorCode : int, errorString : str, advancedOrderRejectJson : str = ""):
        match errorCode:
            case ErrorCodes.HMDS_DATA_FARM:
                self.connection_status.HMDS_DATA_FARM = True
            case ErrorCodes.MARKET_DATA_FARM:
                self.connection_status.MARKET_DATA_FARM = True
            case ErrorCodes.SEC_DEF_DATA_FARM:
                self.connection_status.SEC_DEF_DATA_FARM = True
            case ErrorCodes.HMDS_DATA_FARM_INACTIVE:
                self.connection_status.HMDS_DATA_FARM = True
            case ErrorCodes.CANT_CONNECT_TO_TWS:
                raise NotImplementedError("Failed to connect to TWS")
            case _:
                raise NotImplementedError(f"Error Code: {errorCode} | Error Message: {errorString}")

        if advancedOrderRejectJson:
            logger.error("ERROR %s %s %s %s", reqId, errorCode, errorString, advancedOrderRejectJson)
        else:
            logger.error("ERROR %s %s %s", reqId, errorCode, errorString)

    def connectAck(self):
        super().connectAck()
        logging.warning("Connected to TWS")
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
            timed_out = not self.app.condition.wait(TIMEOUT)
        if timed_out or not self.app.isConnected():
            raise ConnectionError("Failed to connect to TWS")

    def connect(self) -> None:
        logging.warning("Connecting to TWS...")
        self.app.connect(str(self.IP_ADDRESS), self.PORT, self.CLIENT_ID)
        self.api_thread = threading.Thread(
            target=run_loop, args=(self.app,), daemon=True)
        self.api_thread.start()
        self.__await_connection()
        with self.app.connection_status.condition:
            self.app.connection_status.condition.wait(TIMEOUT)
        # while not self.app.connection_status.is_connected:
        #     time.sleep(1)

    def disconnect(self) -> None: self.app.disconnect()

    def __get_account_summary(self, tag : AccountSummaryTag) -> dict[str, list[tuple[str|float|Decimal|int, str]]]:
        with self.app.condition:
            self.app.account_summary = {} # Clear account summary dictionary
            self.app.reqAccountSummary(0, "All", tag)
            timed_out = not self.app.condition.wait(TIMEOUT)
        if timed_out:
            raise TimeoutError(f"Failed to retrieve account summary for {tag}")
        return self.app.account_summary

    def get_soonest_valid_expiry(self, contract : Contract, min_DTE : int) -> str:
        """
        Gets the next expiration date for a contract that is at least min_DTE days away
        """

        # if self.app.connection_status.SEC_DEF_DATA_FARM:

        possible_contracts : dict[str, Contract] = self.get_contract_details(contract)
        expirations : list[str] = [x.lastTradeDateOrContractMonth for x in possible_contracts.values()]
        expirations.sort() #@ Nice thing about YYYYMMDD is that it's sortable

        #* finds the first expiration that meets the condition, else None
        #* ... likely too Pythonic and could be rewritten for better clarity
        expiration_date = next(
            (
                datetime.datetime.strptime(expiration, "%Y%m%d").strftime("%Y%m%d")
                for expiration in expirations
                if datetime.datetime.strptime(expiration, "%Y%m%d").date() >= datetime.date.today() + datetime.timedelta(days=min_DTE)
            ),
            None
        )

        if expiration_date is None:
            raise ValueError(f"Contract {contract} has no valid contract months")

        return expiration_date
 

    def initialize_desired_contracts(self, account : Account, min_DTE) -> None:
        """
        Takes an account object and finds the nearest contracts with at least X days until expiry.
        Where X is account.min_DTE; operates inplace
        """
        for position in account.positions:
            expiration_date : str = self.get_soonest_valid_expiry(position.contract, min_DTE)
            position.contract.lastTradeDateOrContractMonth = expiration_date
            desired_contract : dict[str, Contract] = self.get_contract_details(position.contract)

            if len(desired_contract.values()) == 0:
                raise Exception(f"Contract {position.contract} has no suitable contract months")
            if len(desired_contract.values()) > 1:
                raise Exception(f"Contract {position.contract} has multiple valid contract months, please specify one")

            position.contract = desired_contract.popitem()[1]

    @property
    def initial_margin(self) -> tuple[str, Decimal]:
        initial_margin = self.__get_account_summary(AccountSummaryTag.FULL_INIT_MARGIN_REQ)[AccountSummaryTag.FULL_INIT_MARGIN_REQ][0]
        return {initial_margin[1] : Decimal(initial_margin[0])}

    @property
    def maintenance_margin(self) -> tuple[str, Decimal]:
        maintenance_margin = self.__get_account_summary(AccountSummaryTag.FULL_MAINT_MARGIN_REQ)[AccountSummaryTag.FULL_MAINT_MARGIN_REQ][0]
        return {maintenance_margin[1] : Decimal(maintenance_margin[0])}

    @property
    def cash_balances(self) -> dict[str, Decimal]:
        account_summary = self.__get_account_summary(AccountSummaryTag.LEDGER)
        return {currency: Decimal(value) for tag, values in account_summary.items() if tag == "TotalCashBalance" for value, currency in values}

    @property
    def exchange_rates(self) -> dict[str, Decimal]:
        account_summary = self.__get_account_summary(AccountSummaryTag.LEDGER)
        return {currency: Decimal(value) for tag, values in account_summary.items() if tag == "ExchangeRate" for value, currency in values}

    @property
    def current_positions(self) -> list[Position]:
        with self.app.condition:
            self.app.positions = [] # Clear positions list
            self.app.reqPositions()
            timed_out = not self.app.condition.wait(TIMEOUT)
        if timed_out:
            raise TimeoutError("Failed to retrieve current positions")

        positions : list[Position] = self.app.positions

        #! necessary to get exchange and other pieces of data for the contract
        #* hate this, but contract generated doesn't exist in their own DB
        for position in positions:
            contract_by_ID = Contract(conId=position.contract.conId)
            positions.append(
                Position(
                    contract=self.get_contract_details(contract_by_ID)[position.contract.conId],
                    quantity=position.quantity
                )
            )
        return positions

    def get_contract_details(
            self,
            contract : Contract) -> dict[str, Contract]:
        with self.app.condition:
            self.app.contract_details = {} # Clear contracts dictionary
            self.app.reqContractDetails(0, contract)
            timed_out = not self.app.condition.wait(TIMEOUT)
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
                timed_out = not self.app.condition.wait(TIMEOUT)
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
            trades : list[Trade],
            trading_algorithm : typing.Callable) -> None:
        self.cancel_outstanding_orders()
        for trade in trades:
            logging.warning(f"{trade.order.action} {trade.order.totalQuantity} {trade.contract.symbol} {trade.contract.lastTradeDateOrContractMonth} on {trade.contract.exchange} using {trading_algorithm.__name__}")
            trading_algorithm(self.app, trade.contract, trade.order)
        if WAIT_FOR_TRADES:
            self.__wait_for_trades()

    def __del__(self) -> None:
        """
        Destructor ensures disconnect when picked up by garbage collector
        """
        self.disconnect()

@contextmanager
def api_handler_context(
    IP_ADDRESS : ipaddress.IPv4Address,
    PORT : int,
    CLIENT_ID : int):
    handler = APIHandler(IP_ADDRESS, PORT, CLIENT_ID)
    handler.connect()
    try:
        yield handler
    finally:
        handler.disconnect()
        handler.api_thread.join()

if __name__ == '__main__':
    with api_handler_context(ipaddress.ip_address("127.0.0.1"), 4002, 0) as api_handler:
        api_handler.cancel_outstanding_orders()
        print(api_handler.app.connection_status.is_connected)