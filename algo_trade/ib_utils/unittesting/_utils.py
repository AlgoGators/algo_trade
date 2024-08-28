from src._contract import Contract
from ibapi.order import Order
from decimal import Decimal

def position_is_equal(position1 : dict[str, list[Contract, Decimal]], position2 : dict[str, list[Contract, Decimal]]):
    for i in position1.keys():
        if position1[i][0] != position2[i][0]:
            print("nope")
            return False
        if position1[i][1] != position2[i][1]:
            print("no")
            return False
    return True

def contract_is_equal(contract1 : Contract, contract2 : Contract):
    return contract1 == contract2

def IBKR_positions_is_equal(position1 : dict[Contract, Decimal], position2 : dict[Contract, Decimal]):
    for i in position1.keys():
        if i not in position2.keys():
            return False
        if position1[i] != position2[i]:
            return False
    return True

def trades_is_equal(trades1 : list[tuple[Contract, Order]], trades2 : list[tuple[Contract, Order]]):
    trade_dict1 : dict[Contract, Order]= {}
    trade_dict2 : dict[Contract, Order]= {}
    for i in trades1:
        trade_dict1[i[0]] = i[1]
    for i in trades2:
        trade_dict2[i[0]] = i[1]
    
    if len(trade_dict1) != len(trade_dict2):
        return False
    
    for i in trade_dict1.keys():
        if i not in trade_dict2.keys():
            return False
        if trade_dict1[i].action != trade_dict2[i].action:
            return False
        if trade_dict1[i].totalQuantity != trade_dict2[i].totalQuantity:
            return False
        if trade_dict1[i].orderType != trade_dict2[i].orderType:
            return False
    
    return True