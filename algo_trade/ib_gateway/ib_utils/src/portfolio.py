import datetime
from decimal import Decimal
from ibapi.order import Order
from ..src.api_handler import APIHandler
from ..src._enums import OrderAction
from ..src._contract import Contract

#! Should restructure data so it can be held_portfolio and desired_portfolio
class Portfolio:
    def __init__(
            self,
            api_handler : APIHandler) -> None:
        self.api_handler : APIHandler = api_handler

    def get_expiring_contracts(
            self,
            contracts : dict[str, Contract],
            min_DTE : int) -> dict[str, Contract]:
        expiring_contracts : dict[str, Contract] = {}
        for contractId, contract in contracts.items():
            if contract.lastTradeDateOrContractMonth < datetime.date.today() + datetime.timedelta(days=min_DTE):
                expiring_contracts[contractId] = contract
        return expiring_contracts

    def get_expirations(
            self,
            contract : Contract) -> list[str]:
        possible_contracts : dict[str, Contract] = self.api_handler.get_contract_details(contract)
        return [x.lastTradeDateOrContractMonth for x in possible_contracts.values()]

    def get_next_expiration_date(self, contract : Contract, min_DTE : int) -> str:
        """
        Gets the next expiration date for a contract that is at least min_DTE days away
        """
        expirations : list[str] = self.get_expirations(contract)
        expirations = sorted(expirations) #@ Nice thing about YYYYMMDD is that it's sortable
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

    def get_desired_positions(
            self,
            contracts : dict[Contract, Decimal],
            min_DTE : int) -> dict[str, list[Contract, Decimal]]:
        """
        Gets the desired contracts based on the desired positions and minimum DTE
            ... finds the contract months we want to hold for the desired positions
        """
        desired_contracts : dict[str, list[Contract, Decimal]] = {}

        for contract, position in contracts.items():
            expiration_date : str = self.get_next_expiration_date(contract, min_DTE)
            contract.lastTradeDateOrContractMonth = expiration_date

            desired_contract : dict[str, Contract] = self.api_handler.get_contract_details(contract)

            if len(desired_contract.values()) == 0:
                raise Exception(f"Contract {contract} has no suitable contract months")
            if len(desired_contract.values()) > 1:
                raise Exception(f"Contract {contract} has multiple valid contract months, please specify one")

            desired_contracts[next(iter(desired_contract.keys()))] = [next(iter(desired_contract.values())), position]
        return desired_contracts

    def get_required_trades(
            self,
            held_positions : dict[str, list[Contract, Decimal]],
            desired_positions : dict[str, list[Contract, Decimal]]) -> list[tuple[Contract, Order]]:
        """
        Returns the trades required to get from held_positions to desired_positions
        """
        trades : list[tuple[Contract, Order]] = []

        for contract_id in desired_positions.keys():
            if contract_id not in held_positions:
                held_positions[contract_id] = [desired_positions[contract_id][0], 0]

        for contract_id in held_positions.keys():
            if contract_id not in desired_positions:
                desired_positions[contract_id] = [held_positions[contract_id][0], 0]

        for contract_id, (desired_contract, desired_quantity) in desired_positions.items():
            held_contract, held_quantity = held_positions[contract_id]
            delta = desired_quantity - held_quantity
            if delta != 0:
                order = Order()
                order.action = OrderAction.BUY if delta > 0 else OrderAction.SELL
                order.totalQuantity = abs(delta)
                trades.append((desired_contract, order))

        return trades
