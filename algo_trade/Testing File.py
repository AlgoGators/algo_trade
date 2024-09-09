import pandas as pd
import numpy as np
from datetime import datetime
from pnl import PnL

# Assuming PnL class is already imported

# Load the sample data from the CSV files
positions = pd.read_csv('positions.csv', index_col=0, parse_dates=True)
prices = pd.read_csv('prices.csv', index_col=0, parse_dates=True)
multipliers = pd.read_csv('multipliers.csv', index_col=0)

# Define initial capital
capital = 100000

# Initialize the PnL object
pnl = PnL(positions, prices, capital, multipliers, 0.01)

# Test daily point returns
daily_point_returns = pnl.get(PnL.ReturnType.POINT, PnL.Timespan.DAILY, aggregate=False)
print("Daily Point Returns:\n", daily_point_returns)

# Test cumulative point returns
cumulative_point_returns = pnl.get(PnL.ReturnType.POINT, PnL.Timespan.CUMULATIVE, aggregate=False)
print("Cumulative Point Returns:\n", cumulative_point_returns)

# Test daily percent returns
daily_percent_returns = pnl.get(PnL.ReturnType.PERCENT, PnL.Timespan.DAILY, aggregate=False)
print("Daily Percent Returns:\n", daily_percent_returns)

# Test cumulative percent returns
cumulative_percent_returns = pnl.get(PnL.ReturnType.PERCENT, PnL.Timespan.CUMULATIVE, aggregate=False)
print("Cumulative Percent Returns:\n", cumulative_percent_returns)

# Test Sharpe Ratio
sharpe_ratio = pnl.get_sharpe_ratio()
print("Sharpe Ratio:\n", sharpe_ratio)

# Test Volatility (daily)
daily_volatility = pnl.get_volatility(PnL.Timespan.DAILY)
print("Daily Volatility:\n", daily_volatility)

# Test Volatility (annualized)
annualized_volatility = pnl.get_volatility(PnL.Timespan.ANNUALIZED)
print("Annualized Volatility:\n", annualized_volatility)

# Test mean return (daily)
daily_mean_return = pnl.get_mean_return(PnL.Timespan.DAILY)
print("Daily Mean Return:\n", daily_mean_return)

# Test mean return (cumulative)
cumulative_mean_return = pnl.get_mean_return(PnL.Timespan.CUMULATIVE)
print("Cumulative Mean Return (CAGR):\n", cumulative_mean_return)


print("Points-based Drawdown:\n", pnl.drawdown())

print("Turnover (number of positions traded):\n", pnl.get_turnover())

print("Transaction costs:\n", pnl.get_transaction_costs())

print(pnl.information_ratio())
