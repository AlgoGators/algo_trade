import logging
import ipaddress
import pandas as pd
from .src.trading_algorithm import TradingAlgorithm
from .src.api_handler import APIHandler
from .src.portfolio import Portfolio
from .src.data_interface import DataInterface
from .src._config import LOCALHOST, PORT, CLIENT_ID
from .src._enums import AdaptiveOrderPriority
from .src import _error_handler

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s")

def update_portfolio(data_positions : dict[str, float], instruments_df : pd.DataFrame, orderType : AdaptiveOrderPriority, min_DTE) -> None:
    IBKR_positions = DataInterface(instruments_df, data_positions=data_positions).IBKR_positions

    api_handler = APIHandler(
        IP_ADDRESS=ipaddress.ip_address(LOCALHOST),
        PORT=PORT,
        CLIENT_ID=CLIENT_ID
    )

    api_handler.connect()

    api_handler.cancel_outstanding_orders()

    held_positions = api_handler.get_current_positions()

    portfolio = Portfolio(api_handler)
    desired_instruments = portfolio.get_desired_positions(IBKR_positions, min_DTE)

    trades = portfolio.get_required_trades(held_positions, desired_instruments)

    api_handler.place_orders(
        trades, TradingAlgorithm(orderType).adaptive_market_order)

    api_handler.disconnect()

if __name__ == "__main__":
    data_positions = {
        "ES": 1,
        "NQ": 0,
        "FDAX" : 0,
    }

    instruments_df = pd.read_csv("ib_utils/unittesting/instruments.csv")

    update_portfolio(data_positions, instruments_df)
