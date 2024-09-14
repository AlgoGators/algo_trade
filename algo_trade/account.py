import pandas as pd
import datetime
from decimal import Decimal
import dataclasses

from algo_trade.ib_utils.src._contract import Contract
from algo_trade.ib_utils.src.api_handler import APIHandler
from algo_trade.ib_utils.src._enums import OrderAction
from ibapi.order import Order

@dataclasses.dataclass
class Position():
    contract : Contract
    quantity : Decimal

@dataclasses.dataclass
class Trade():
    contract : Contract
    order : Order

class Account:
    def __init__(
            self,
            positions : list[Position],
            min_DTE : int = -1) -> None:
        self.positions = positions
        self.min_DTE = min_DTE

    def __sub__(self, held : 'Account') -> list[Trade]:
        if not isinstance(held, Account):
            raise ValueError("Can only subtract Account from Account")
        
        desired_IDs = [position.contract.conId for position in self.positions]
        held_IDs = [position.contract.conId for position in held.positions]

        #* These two blocks ensure that all contracts supplied have an expiry, effectively
        if None in desired_IDs:
            raise ValueError("At least one desired position lacks a contract ID definition")
        if None in held_IDs:
            raise ValueError("At least one held position lacks a contract ID definition")

        trades : list[Trade] = []

        #* add 0 values for the contracts that we don't have in desired positions
        for held_position in held.positions:
            if held_position.contract.conId not in desired_IDs:
                zero_position = dataclasses.replace(held_position) #! NEED to make a copy
                zero_position.quantity = Decimal()
                self.positions.append(zero_position)

        for desired_position in self.positions:
            delta : Decimal
            if desired_position.contract.conId not in held_IDs:
                delta = desired_position.quantity
            else:
                held_quantity = held.positions[held_IDs.index(desired_position.contract.conId)].quantity
                delta = desired_position.quantity - held_quantity

            # No trade needed
            if delta == 0:
                continue

            order = Order()
            order.action = OrderAction.BUY if delta > 0 else OrderAction.SELL
            order.totalQuantity = abs(delta)
            trades.append(Trade(desired_position.contract, order))

        return trades


if __name__ == "__main__":
    held_positions = [
        Position(Contract('ESZ24'), Decimal(1)),
        Position(Contract('NQZ24'), Decimal(-1))
    ]
    held_account = Account(held_positions, 5)
    new_positions = [
        Position(Contract('ESZ24'), Decimal(2)),
        Position(Contract('YMZ24'), Decimal(-2))
    ]
    new_account = Account(new_positions, 5)

    trades = new_account - held_account
    print(trades)