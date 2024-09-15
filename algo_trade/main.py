"""
Main module for the algo_trade package.

The main module for the algo_trade package. This module contains the main function that is executed when the package is run as a script.

The function includes the collection of all the instruments, the creation of the a portfolio holding our strategy, and the optimization
"""

import pandas as pd # type: ignore
from pathlib import Path
from functools import partial
import asyncio
from algo_trade.dyn_opt import dyn_opt
from algo_trade.rules import equal_weight, capital_scaling, risk_parity, trend_signals, IDM
from algo_trade.trading_system import TradingSystem
from algo_trade.ib_utils.account import Account
from algo_trade.ib_utils._enums import AdaptiveOrderPriority
from algo_trade.update_account import update_account
from algo_trade.instrument import Future, initialize_instruments, SecurityType, fetch_futures_data
from algo_trade.pnl import PnL
from algo_trade.strategy import Strategy, FutureDataFetcher
from algo_trade.risk_measures import RiskMeasure, CRV
from algo_trade.risk_limits import portfolio_multiplier, position_limit

class TrendFollowing(Strategy[Future]):
    def __init__(self, instruments: list[Future], risk_object: RiskMeasure, capital: float):
        super().__init__(capital=capital)
        self.instruments: list[Future] = instruments
        self.risk_object = risk_object
        self.rules = [
            partial(risk_parity, risk_object=self.risk_object),
            partial(trend_signals, instruments=instruments, risk_object=self.risk_object),
            partial(equal_weight, instruments=instruments),
            partial(capital_scaling, instruments=instruments, capital=capital),
            partial(IDM, risk_object=self.risk_object)
        ]
        self.scalars = []
        self.fetch_data()  # Fetch the data for the instruments

    def fetch_data(self) -> None:
        """
        The Fetch data method for the Trend Following strategy is requires the following instrument specific data:
        1. Prices(Open, High, Low, Close, Volume)
        2. Backadjusted Prices (Close)
        """
        # Load the front calendar contract data with a daily aggregation
        FutureDataFetcher.fetch_front(self.instruments)

### Example TradingSystem
class Trend(TradingSystem):
    def __init__(self, instruments : list[Future], risk_target : float, capital : float):
        super().__init__()
        self.risk_object = CRV(
            risk_target=risk_target,
            instruments=instruments,
            window=100,
            span=32
        )
        self.weighted_strategies = [
            (1.0, TrendFollowing(instruments, self.risk_object, capital))
        ]
        self.capital = capital
        self.instruments = instruments
        self.trading_system_rules = [
            partial(
                dyn_opt,
                instrument_weights=equal_weight(instruments=instruments),
                cost_per_contract=3.0,
                asymmetric_risk_buffer=0.05, 
                cost_penalty_scalar=10, 
                position_limit_fn=position_limit(
                    max_leverage_ratio=2.0,
                    minimum_volume=100,
                    max_forecast_ratio=2.0,
                    max_forecast_buffer=0.5,
                    IDM=2.5,
                    tau=risk_target),
                portfolio_multiplier_fn=portfolio_multiplier(
                    max_portfolio_leverage=20,
                    max_correlation_risk=0.70,
                    max_portfolio_volatility=0.40,
                    max_portfolio_jump_risk=0.80)
            )
        ]

def main() -> None:
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
    contract_path : Path = Path("data/contract.csv")
    if not contract_path.exists():
        raise FileNotFoundError(f"Contract file not found at {contract_path}")

    # Load the instruments
    instruments_dataframe : pd.DataFrame = pd.read_csv(contract_path)

    instruments_dataframe.dropna(subset=['dataSet'], inplace=True)

    # Initialize the instruments
    futures : list[Future] = [future for future in initialize_instruments(instruments_dataframe) if future.security_type == SecurityType.FUTURE]

    asyncio.run(fetch_futures_data(futures, rate=5))

    # If future.front is None, drop from the list
    for future in futures:
        try:
            if future.front is None:
                futures.remove(future)
        except:
            futures.remove(future)

    # Create a portfolio
    trend : TradingSystem = Trend(
        instruments=futures,
        risk_target=0.20,
        capital=1_000_000.0
    )

    account : Account = trend[-1]

    update_account(
        account=account,
        order_priority=AdaptiveOrderPriority.NORMAL
    )

if __name__ == "__main__":
    main()
