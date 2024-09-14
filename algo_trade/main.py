"""
Main module for the algo_trade package.

The main module for the algo_trade package. This module contains the main function that is executed when the package is run as a script.

The function includes the collection of all the instruments, the creation of the a portfolio holding our strategy, and the optimization
"""

from algo_trade.strategy import Strategy
from algo_trade.trading_system import TradingSystem
from algo_trade.implementations import TrendFollowing
from algo_trade.instrument import Future, RollType, ContractType, Agg, initialize_instruments
from algo_trade.pnl import PnL
import pandas as pd
from pathlib import Path


def main():
    """
    Main function for the algo_trade package

    The main function constructs our instruments, creates a portfolio, finds our positions, and optimizes our portfolio.
    The main function is the entry point for the algo_trade package and is executed when the package is run as a script.
    Args:
    None

    Returns:
    None
    """

    # Find the list of instruments to trade
    contract_path: Path = Path("data/contract.csv")
    if not contract_path.exists():
        raise FileNotFoundError(f"Contract file not found at {contract_path}")

    # Load the instruments
    instruments_dataframe: pd.DataFrame = pd.read_csv(contract_path)

    # Initialize the instruments
    instruments: list[Future] = initialize_instruments()
    
    # Create a portfolio
    trading_system = TradingSystem(
        strategies=[
            (1.0, TrendFollowing(instruments, risk_target=0.05, capital=1000000))
        ]
    )