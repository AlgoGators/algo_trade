import sys

# Add parent directory to path
sys.path.append('../ib_utils')

import unittest
import pandas as pd
from decimal import Decimal

from src._contract import Contract
from src.data_interface import DataInterface
from unittesting._utils import IBKR_positions_is_equal

class TestDataInterface(unittest.TestCase):
    instruments_df = pd.read_csv('unittesting/instruments.csv')

    def test_IBKR_positions(self):
        data_positions = {'ES' : 1, 'NQ' : 2}
        contractA, contractB = Contract(), Contract()
        contractA.symbol, contractA.exchange, contractA.multiplier, contractA.currency, contractA.secType = 'MES', 'CME', 5, 'USD', 'FUT'
        contractB.symbol, contractB.exchange, contractB.multiplier, contractB.currency, contractB.secType = 'MNQ', 'CME', 2, 'USD', 'FUT'
        expected_positions = {contractA : Decimal(1), contractB : Decimal(2)} # String representation of contracts
        data_interface = DataInterface(self.instruments_df, data_positions=data_positions)
        ibkr_positions = data_interface.IBKR_positions
        self.assertTrue(IBKR_positions_is_equal(expected_positions, ibkr_positions))
            
    def test_data_positions(self):
        contractA, contractB = Contract(), Contract()
        contractA.symbol, contractA.exchange, contractA.multiplier, contractA.currency = 'MES', 'CME', 5, 'USD'
        contractB.symbol, contractB.exchange, contractB.multiplier, contractB.currency = 'DAX', 'EUREX', 1, 'EUR'
        ibkr_positions = {contractA: 1, contractB: 2}
        expected_positions = {'ES': Decimal(1), 'FDAX': Decimal(2)}
        data_interface = DataInterface(self.instruments_df, IBKR_positions=ibkr_positions)
        self.assertEqual(expected_positions, data_interface.data_positions)
    
    def test_none_positions(self):
        data_interface = DataInterface(self.instruments_df)
        self.assertEqual({}, data_interface.data_positions)
        self.assertEqual({}, data_interface.IBKR_positions)
    
    def test_both_positions(self):
        contractA, contractB = Contract(), Contract()
        contractA.symbol, contractA.exchange, contractA.multiplier, contractA.currency = 'MES', 'CME', 5, 'USD'
        contractB.symbol, contractB.exchange, contractB.multiplier, contractB.currency = 'DAX', 'EUREX', 1, 'EUR'
        ibkr_positions = {contractA: Decimal(1), contractB: Decimal(2)}
        data_positions = {'ES': 1, 'FDAX': 2}
        with self.assertRaises(ValueError):
            DataInterface(self.instruments_df, IBKR_positions=ibkr_positions, data_positions=data_positions)

if __name__ == '__main__':
    unittest.main(failfast=True)
