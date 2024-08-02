"""

Outline:
...
Calculate risk values using             risk_measures.py
...
Calculate optimized positions using     dyn_opt.py
... (if applicable)
Place orders using                      update_portfolio.py

"""

import importlib
import json
import os
import sys
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import subprocess

from algo_trade.data_engine.pipeline import Pipeline
from algo_trade.risk_management.dyn_opt.dyn_opt import aggregator
from algo_trade.risk_management.risk_measures.risk_functions import get_jump_covariances
from algo_trade.risk_management.risk_measures.risk_measures import RiskMeasures
from algo_trade.ib_gateway.ib_utils.update_portfolio import update_portfolio
from algo_trade.ib_gateway.ib_utils.src._enums import AdaptiveOrderPriority

# Add the ib_utils directory to the Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "ib-gateway/ib_utils"))
)

# Add the ib_utils/src directory to the Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "ib-gateway/ib_utils/src"))
)

class SETTINGS:
    weights = (0.01, 0.01, 0.98)
    warmup = 100
    unadj_column = "Close Unadjusted"
    expiration_column = "Expiration"
    date_column = "Timestamp"
    capital = 500_000
    fixed_cost_per_contract = 3.00
    tau = 0.20
    asymmetric_risk_buffer = 0.05
    instrument_weight = 0.10
    IDM = 2.5
    maxmimum_forecast_ratio = 2.0
    max_acceptable_pct_of_open_interest = 0.01
    max_forecast_buffer = 0.5
    maximum_position_leverage = 2.0
    maximum_portfolio_leverage = 20.0
    maximum_correlation_risk = 0.65
    maximum_portfolio_risk = 0.30
    maximum_jump_risk = 0.75
    cost_penalty_scalar = 10
    DOCKER_PATH = "algo_trade/ibc/docker-compose.yml"

    ORDER_PRIORITY = AdaptiveOrderPriority.NORMAL
    MIN_DTE = 5

def merge_dfs(d : dict[str, pd.Series]) -> pd.DataFrame:
    merged_df : pd.DataFrame = pd.DataFrame()
    for key, df in d.items():
        df.name = key
        if merged_df.empty:
            merged_df = df
        else:
            merged_df = pd.concat([merged_df, df], axis=1, join="outer")
    return merged_df

class Docker:
    def up():
        subprocess.run(["docker-compose", "-f", SETTINGS.DOCKER_PATH, "up", "-d"], shell=True)
    def down():
        subprocess.run(["docker-compose", "-f", SETTINGS.DOCKER_PATH, "down"], check=True)

def main():
    """
    Order of Operations:
    1. Start the docker container

    2. Data Engine:
        - Rebuilding our data store using the pipeline
        - Transforming the data
        - @TODO: Load to the database
    3. Risk Measures Calculation:
        - Calculate our cov and var matrices
            - Returns (weekly and daily)
            - GARCH variances
            - GARCH covariances
    4. Dynamic Optimization:
        - Position beep bop boop
    5. Place orders
    6. Close docker
    """
    try:
        Docker.up()

        # 2. Data Engine
        pipeline: Pipeline = Pipeline()
        # pipeline.rebuild()
        pipeline.transform()
        # pipeline.load()

        # Get the data
        price_tables: Dict[str, pd.DataFrame] = pipeline.get_price_tables()

        # 3. Risk Measures Calculation
        risk_measures = RiskMeasures(
            price_tables,
            SETTINGS.weights,
            SETTINGS.warmup,
            SETTINGS.unadj_column,
            SETTINGS.expiration_column,
            SETTINGS.date_column,
        )

        risk_measures.construct()

        daily_returns = risk_measures.daily_returns
        garch_variances = risk_measures.GARCH_variances
        garch_covariances = risk_measures.GARCH_covariances

        jump_covariances = get_jump_covariances(garch_covariances, 0.99, 256)

        instruments_df = pd.read_csv("data/contract.csv")

        multipliers = instruments_df[["dataSymbol", "multiplier"]].set_index("dataSymbol").transpose()

        ideal_positions: pd.DataFrame = pipeline.positions(capital=SETTINGS.capital, tau=SETTINGS.tau, multipliers=multipliers, variances=garch_variances, IDM=SETTINGS.IDM)
        unadj_prices = pipeline.get_prices()
        open_interest = pipeline.get_open_interest()
        unadj_prices = merge_dfs(unadj_prices)
        open_interest = merge_dfs(open_interest)

        instrument_weights = pd.DataFrame(1 / len(ideal_positions.columns), index=ideal_positions.index, columns=ideal_positions.columns)

        # 4. Dynamic Optimization
        positions : pd.DataFrame = aggregator(
            SETTINGS.capital,
            SETTINGS.fixed_cost_per_contract,
            SETTINGS.tau,
            SETTINGS.asymmetric_risk_buffer,
            unadj_prices,
            multipliers,
            ideal_positions,
            garch_covariances,
            jump_covariances,
            open_interest,
            instrument_weights,
            SETTINGS.IDM,
            SETTINGS.maxmimum_forecast_ratio,
            SETTINGS.max_acceptable_pct_of_open_interest,
            SETTINGS.max_forecast_buffer,
            SETTINGS.maximum_position_leverage,
            SETTINGS.maximum_portfolio_leverage,
            SETTINGS.maximum_correlation_risk,
            SETTINGS.maximum_portfolio_risk,
            SETTINGS.maximum_jump_risk,
            SETTINGS.cost_penalty_scalar,
        )

        # 5. Place Orders

        # Assumes ideal_positions is a dataframe, converts last row to dict
        positions_dict = positions.iloc[-1].to_dict()

        update_portfolio(positions_dict, instruments_df, SETTINGS.ORDER_PRIORITY, SETTINGS.MIN_DTE)

    finally:
        Docker.down()

if __name__ == "__main__":
    main()
