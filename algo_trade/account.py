import pandas as pd
import datetime
from decimal import Decimal
from dataclasses import dataclass

from algo_trade.ib_utils.src._contract import Contract
from algo_trade.ib_utils.src.api_handler import APIHandler
from ibapi.order import Order

@dataclass
class Position():
    contract : Contract
    quantity : float

@dataclass
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

    def __sub__(self, other : 'Account') -> list[Trade]:
        if not isinstance(other, Account):
            raise ValueError("Can only subtract Account from Account")
        
        desired_IDs = [position.contract.conId for position in self.positions]
        held_IDs = [position.contract.conId for position in other.positions]

        if None in desired_IDs:
            raise ValueError("At least one desired position lacks a contract ID definition")

        if None in held_IDs:
            raise ValueError("At least one held position lacks a contract ID definition")

        for desired_ID in desired_IDs:
            if 






































# if __name__ == "__main__":
#     dct = {
#         'ES' : 1,
#         'NQ' : -2
#     }

#     df = pd.DataFrame(dct, index=[0])

#     account = Account(None)
#     account.positions = 1
#     print(account.positions)