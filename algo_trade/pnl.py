from enum import Enum, auto
import pandas as pd # type: ignore
import numpy as np
from algo_trade._constants import DAYS_IN_YEAR
import matplotlib.pyplot as plt
from typing import Optional

class PnL:
    """
    PnL class to handle profit and loss calculations
    """
    # Nested ReturnType Enum to define point and percent return types
    class ReturnType(Enum):
        POINT = auto()
        PERCENT = auto()

    # Nested Timespan Enum to define daily, annualized, and cumulative timespans
    class Timespan(Enum):
        DAILY = auto()
        ANNUALIZED = auto()
        CUMULATIVE = auto()

    # Constructor for the PnL class
    def __init__(
            self,
            positions: pd.DataFrame,
            prices: pd.DataFrame,
            capital: float,
            multipliers: pd.DataFrame | None = None,
            transaction_cost_rate: float = 0.0
        ) -> None:
        self.__positions = positions
        self.__prices = prices
        self.__multipliers = multipliers if multipliers is not None else pd.DataFrame(columns=positions.columns, data=np.ones((1, len(positions.columns))))
        self.__capital = capital
        self.__transaction_cost_rate = transaction_cost_rate
        self.__point_returns = self.__get_point_returns()
        self.__turnover = self.__get_turnover()
        self.__transaction_costs = self.__get_transaction_costs()
        self.__benchmark = self.__get_benchmark()

    def __get_benchmark(self) -> pd.Series:
        # Daily percent returns for each instrument
        percent_returns = self.__prices.pct_change().fillna(0)

        # Average daily return (buy and hold assumption: equally weighted)
        benchmark_returns = percent_returns.mean(axis=1)

        return benchmark_returns

    def __get_turnover(self) -> pd.DataFrame:
        """
        Get the turnover (absolute changes in positions)
        """
        # Calculate turnover as absolute change in positions across instruments
        turnover = self.__positions.diff().abs().fillna(0)
        return turnover

    def __get_transaction_costs(self) -> pd.DataFrame:
        """
        Gets transaction costs based on turnover and transaction cost rate
        """
        # Transaction costs based on turnover and transaction cost rate
        transaction_costs = self.__turnover * self.__transaction_cost_rate
        return transaction_costs

    def get_turnover(self, aggregate: bool = True) -> pd.Series | pd.DataFrame:
        """
        Returns turnover with aggregation option
        """
        if aggregate:
            # Sum turnover across all instruments
            return self.__turnover.sum(axis=1)
        else:
            # Return turnover per instrument
            return self.__turnover

    def get_transaction_costs(self, aggregate: bool = True, transaction_cost_rate: Optional[float] = None) -> pd.Series | pd.DataFrame:
        """
        Returns transaction costs with aggregation option
        """
        if transaction_cost_rate != None:
            # Transaction costs based on turnover and transaction cost rate
            transaction_costs = self.__turnover * transaction_cost_rate
            return transaction_costs
        else:
            if aggregate:
                # Sum transaction costs across all instruments
                return self.__transaction_costs.sum(axis=1)
            else:
                # Return transaction costs per instrument
                return self.__transaction_costs

    def get(self, return_type: ReturnType, timespan: Timespan, aggregate: bool = True) -> pd.DataFrame:
        """
        Gets returns based on return type, timespan, and aggregation
        """
        # Match-case to determine behavior based on return_type and timespan
        match return_type, timespan:
            #@ Daily point return
            case self.ReturnType.POINT, self.Timespan.DAILY:
                return self.__point_returns.sum(axis=1) if aggregate else self.__point_returns
            #@ Cumulative point return
            case self.ReturnType.POINT, self.Timespan.CUMULATIVE:
                return self.__point_returns.cumsum().sum(axis=1) if aggregate else self.__point_returns.cumsum()

            #@ Daily percent return
            case self.ReturnType.PERCENT, self.Timespan.DAILY:
                return self.__point_returns / self.__prices.shift(
                    1) if not aggregate else self.__portfolio_percent_returns(self.__capital)
            #@ Cumulative percent return
            case self.ReturnType.PERCENT, self.Timespan.CUMULATIVE:
                return (self.__point_returns / self.__prices.shift(
                    1) + 1).cumprod() - 1 if not aggregate else self.__point_returns.sum(
                    axis=1).cumsum() / self.__capital

            #@ Handle unsupported combinations
            case _:
                raise NotImplementedError(
                    f"The Enums provided or the combination of them: {return_type, timespan}, has not been implemented.")

    def get_sharpe_ratio(self, aggregate: bool = True) -> pd.Series:
        """
        Gets the Sharpe ratio
        """
        returns = self.get(self.ReturnType.PERCENT, self.Timespan.DAILY, aggregate)
        return returns.mean() / returns.std() * DAYS_IN_YEAR ** 0.5

    def get_volatility(self, timespan: Timespan, aggregate: bool = True) -> pd.Series:
        """
        Gets volatility based on timespan and aggregation
        """
        returns = self.get(self.ReturnType.PERCENT, self.Timespan.DAILY, aggregate)
        if timespan == self.Timespan.DAILY:
            return returns.std()
        elif timespan == self.Timespan.ANNUALIZED:
            return returns.std() * DAYS_IN_YEAR ** 0.5
        else:
            raise NotImplementedError(f"The Enum provided: {timespan}, has not been implemented.")

    def get_mean_return(self, timespan: Timespan, aggregate: bool = True) -> pd.Series:
        """
        Gets mean return based on timespan and aggregation
        """
        if timespan == self.Timespan.DAILY:
            returns = self.get(self.ReturnType.PERCENT, self.Timespan.DAILY, aggregate)
            return returns.mean()
        elif timespan == self.Timespan.CUMULATIVE:
            returns = self.get(self.ReturnType.PERCENT, self.Timespan.CUMULATIVE, aggregate)
            total_return = returns.iloc[-1]
            cagr = (1 + total_return) ** (1 / (returns.count() / DAYS_IN_YEAR)) - 1
            return cagr
        raise NotImplementedError(f"The Enum provided: {timespan}, has not been implemented.")

    def drawdown(self, aggregate: bool = True) -> pd.Series: #Currently in points,
        portfolio_returns = self.get(self.ReturnType.POINT, self.Timespan.DAILY, aggregate)

        cumulative_returns = self.__capital + portfolio_returns.cumsum()

        running_max = cumulative_returns.cummax()

        # Calculate drawdown as the drop from the running maximum
        drawdown = (running_max - cumulative_returns) / running_max

        # Replace NaN and infinite values (due to initial period calculations) with 0
        drawdown = drawdown.fillna(0).replace([np.inf, -np.inf], 0)

        return drawdown

    def plot(self, metrics: Optional[list[str]] = None, aggregate: bool = True) -> None:
        if metrics is None:
            metrics = ['returns', 'cumulative', 'drawdown']

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        if 'returns' in metrics:
            returns = self.get(self.ReturnType.PERCENT, self.Timespan.DAILY, aggregate)
            ax.plot(returns.index, returns, label='Daily Returns', color='blue', alpha=0.7)

        # Plot Cumulative Returns
        if 'cumulative' in metrics:
            cumulative_returns = self.get(self.ReturnType.PERCENT, self.Timespan.CUMULATIVE, aggregate)
            ax.plot(cumulative_returns.index, cumulative_returns, label='Cumulative Returns', color='green', alpha=0.7)

            benchmark_cumulative_returns = (self.__benchmark + 1).cumprod() - 1
            ax.plot(benchmark_cumulative_returns.index, benchmark_cumulative_returns,
                    label='Cumulative Returns (Benchmark)', color='orange', alpha=0.7)

        # Plot Drawdown
        if 'drawdown' in metrics:
            drawdown = -self.drawdown(aggregate)
            ax.plot(drawdown.index, drawdown, label='Drawdown (%)', color='red', alpha=0.7)

        # Set title and labels
        ax.set_title("Portfolio Metrics Over Time", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Percentage", fontsize=12)

        # Show the legend
        ax.legend()

        # Display the plot
        plt.grid(True)
        plt.show()

    def tracking_error(self, other: pd.Series) -> auto():
        """
        Calculate tracking error with another series
        """
        tracking_error = np.std(other - self.__benchmark)
        return tracking_error * DAYS_IN_YEAR ** 0.5

    def get_skewness(self, aggregate: bool = True) -> pd.Series:
        """
        Calculate skewness
        """
        returns = self.get(self.ReturnType.PERCENT, self.Timespan.DAILY, aggregate)
        return ((returns - returns.mean()) ** 3).mean() / (returns.std() ** 3)

    def get_kurtosis(self, aggregate: bool = True) -> pd.Series:
        """
        Calculate kurtosis
        """
        returns = self.get(self.ReturnType.PERCENT, self.Timespan.DAILY, aggregate)
        return ((returns - returns.mean()) ** 4).mean() / (returns.std() ** 4)

    def get_max_drawdown(self, aggregate: bool = True) -> pd.Series:
        """
        Calculate max drawdown
        """
        drawdown = self.drawdown(aggregate)
        return drawdown.max()

    def get_calmar_ratio(self, aggregate: bool = True) -> pd.Series:
        """
        Calculate Calmar ratio
        """
        returns = self.get(self.ReturnType.PERCENT, self.Timespan.DAILY, aggregate) * DAYS_IN_YEAR ** 0.5
        max_drawdown = self.get_max_drawdown(aggregate)
        return returns.mean() / max_drawdown

    def get_information_ratio(self, aggregate: bool = True) -> pd.Series:
        """
        Calculate information ratio
        """
        portfolio_returns = self.get(self.ReturnType.PERCENT, self.Timespan.DAILY, aggregate)
        tracking_error = self.tracking_error(portfolio_returns)
        excess_return = (portfolio_returns.mean() - self.__benchmark.mean()) * np.sqrt(DAYS_IN_YEAR)
        return excess_return / tracking_error

    def get_tail_ratio(self, aggregate: bool = True) -> pd.Series:
        """
        Calculate Tail Ratio (95th percentile return / 5th percentile return)
        """
        returns = self.get(self.ReturnType.PERCENT, self.Timespan.DAILY, aggregate)
        return np.percentile(returns, 95) / np.abs(np.percentile(returns, 5))

    def __portfolio_percent_returns(self, capital: float) -> pd.Series:
        """
        Helper method to calculate portfolio percent returns based on capital
        """
        capital_series = pd.Series(data=capital, index=self.__point_returns.index) + self.__point_returns.sum(axis=1)
        return capital_series / capital_series.shift(1) - 1

    def __get_point_returns(self) -> pd.DataFrame:
        """
        Helper method to calculate point returns
        """
        pnl = pd.DataFrame()
        for instrument in self.__positions.columns:
            pos_series = self.__positions[instrument]

            # Merge positions and prices
            both_series = pd.concat([pos_series, self.__prices[instrument]], axis=1)
            both_series.columns = ["positions", "prices"]

            # Forward fill prices only, avoid forward-filling positions
            both_series['prices'] = both_series['prices'].ffill()

            # Calculate price returns
            price_returns = both_series.prices.diff()

            # Shift positions to reflect previous day's holding and multiply by price return
            returns = both_series.positions.shift(1) * price_returns

            # Handle NaNs in returns
            returns.fillna(0.0, inplace=True)

            # Apply multipliers to the returns
            pnl[instrument] = returns * self.__multipliers[instrument].iloc[0]
        return pnl
