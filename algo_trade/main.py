"""
Main module for the algo_trade package.

The main module for the algo_trade package. This module contains the main function that is executed when the package is run as a script.

The function includes the collection of all the instruments, the creation of the a portfolio holding our strategy, and the optimization
"""

from algo_trade.strategy import Strategy
from algo_trade.portfolio import Portfolio
from algo_trade.implementations import TrendFollowing
from algo_trade.instrument import Future, RollType, ContractType, Agg, initialize_instruments
from algo_trade.pnl import PnL


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

    # Initialize the instruments
    instruments: list[Future] = initialize_instruments()
    
    # Create a portfolio
    portfolio = Portfolio(
        strategies=[
            (1.0, TrendFollowing(instruments, risk_target=0.05, capital=1000000))
        ]
    )
    
    # Find the positions
    portfolio.find_positions()
    
    # Optimize the portfolio
    portfolio.optimize()
    
    # Calculate the PnL
    pnl = PnL(portfolio=portfolio)
    pnl.calculate()
    
    # Print the PnL
    print(pnl)

