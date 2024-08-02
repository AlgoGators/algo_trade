"""
Author: Cole Rottenberg
Description: This file contains the Transformation class which is used to backadjust the raw data, and find other key metrics.
"""

import os

import numpy as np
import pandas as pd
import tqdm

from .base import Carry, Trend
from .std_daily_price import standardDeviation

base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "data", "raw")
processed_dir = os.path.join(base_dir, "data", "processed")


class Transformation:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.path = os.path.join(data_dir, f"{self.symbol}_full.parquet")
        self.data_path = os.path.join(data_dir, f"{self.symbol}_data.parquet")
        self.def_path = os.path.join(data_dir, f"{self.symbol}_definitions.parquet")

        # Open the parquet file
        self.open()
        # Group the data by front and back month
        self.group()
        # Perform backadjustment
        self.backadjust()

        # Find volatility
        self.stdDev: standardDeviation = self.get_risk()

    # Getters
    def get_symbol(self) -> str:
        return self.symbol

    def get_raw(self) -> pd.DataFrame:
        return self.raw

    def get_backadjusted(self) -> pd.DataFrame:
        return self.backadjusted

    def get_front(self) -> pd.DataFrame:
        return self.front

    def get_back(self) -> pd.DataFrame:
        return self.back

    def get_risk(self) -> standardDeviation:
        """
        The get_risk method is used to find the standard deviation of the backadjusted data.

        Args:
        - None

        Returns:
        - The standard deviation of the backadjusted data 
        """
        data = self.format_trend()
        return standardDeviation(adjusted_price=data["Close"], current_price=data["Close Unadjusted"])

    def get_current_price(self) -> pd.Series:
        """
        The get_current_price method is used to find the current price of the data.

        Args:
        - None

        Returns:
        - pd.Series: The current price of the data
        """
        return self.format_trend()["Close Unadjusted"]

    def get_open_interest(self) -> pd.Series:
        """
        The get_open_interest method is used to find the open interest of the data.

        Args:
        - None

        Returns:
        - pd.Series: The open interest of the data
        """
        return self.format_trend()["Volume"]

    def open(self):
        try:
            self.raw = pd.read_parquet(self.path)
            self.raw_data = pd.read_parquet(self.data_path)
            self.raw_definitions = pd.read_parquet(self.def_path)
        except FileNotFoundError:
            print(f"Error: {self.symbol} has no raw data... failed to open")

    def group(self):
        # WARNING: If Roll Rule changes, this will need to be updated to reflect the new roll rule
        roll = ["c.0", "c.1"]
        # Group the raw data by front month and back month
        # self.grouped = self.raw.groupby("symbol_data")
        data = self.raw_data.copy()
        defintions = self.raw_definitions.copy()
        front_data = data[data["symbol"] == f"{self.symbol}.{roll[0]}"].copy(deep=True)
        back_data = data[data["symbol"] == f"{self.symbol}.{roll[1]}"].copy(deep=True)
        front_definitions = defintions[
            defintions["symbol"] == f"{self.symbol}.{roll[0]}"
        ].copy(deep=True)
        back_definitions = defintions[
            defintions["symbol"] == f"{self.symbol}.{roll[1]}"
        ].copy(deep=True)
        front = pd.merge_asof(
            front_data.sort_index(),
            front_definitions.sort_index(),
            direction="nearest",
            on="timestamp",
            suffixes=("_data", "_definition"),
        )
        back = pd.merge_asof(
            back_data.sort_index(),
            back_definitions.sort_index(),
            direction="nearest",
            on="timestamp",
            suffixes=("_data", "_definition"),
        )

        """
        We have run into the issue that there are duplicates within the font and back month dataframes and one of the expirations is incorrect.
        1. We find all the duplicates in the front and back month dataframes
        2. We sort the duplicates by the key_0 column to just look at the first two duplicates
        3. Assuming that the order of duplicates stays consistent, we determine if the incorrect duplicate comes first or second
        4. We drop the incorrect duplicate
        LOGIC:
        If the first duplicate has a smaller difference between the expiration and the current date(key_0) than the second duplicate, we keep it for the front month dataframe
        If the first duplicate has a larger difference between the expiration and the current date(key_0) than the second duplicate, we keep the second duplicate for the front month dataframe
        AND vice versa for the back month dataframe.
        """

        # * Front Month
        # Group each duplicate row together by groupby each date in key_0
        # front["diff"] = front["expiration"] - front["timestamp"]

        # front.reset_index(inplace=True)

        # def handle_min(group):
        #     min_diff = group["diff"].idxmin()
        #     return group.loc[[min_diff]]

        # front = front.groupby("timestamp").apply(handle_min).reset_index(drop=True)

        # front.set_index("timestamp", inplace=True)
        # front.drop(columns=["diff"], inplace=True)

        # * Back Month

        # Group each duplicate row together by groupby each date in key_0
        # back["diff"] = back["expiration"] - back["timestamp"]
        # back.reset_index(inplace=True)

        # def handle_max(group):
        #     max_diff = group["diff"].idxmax()
        #     return group.loc[[max_diff]]

        # back = back.groupby("timestamp").apply(handle_max).reset_index(drop=True)

        # back.set_index("timestamp", inplace=True)
        # back.drop(columns=["diff"], inplace=True)

        # Store the front and back month dataframes
        self.front = front.copy(deep=True)
        self.back = back.copy(deep=True)

    def backadjust(self):
        # Backadjust the raw dataframe

        # Flip the dataframe to backadjust
        back = self.front[::-1].copy()

        # Adding columns to the dataframe
        back["open_adj"] = back["open"]
        back["high_adj"] = back["high"]
        back["low_adj"] = back["low"]
        back["close_adj"] = back["close"]
        # Add Adjustment Factor Column
        back["adj_factor"] = 0.0

        adj_factor = 0.0
        # Loop through the dataframe and adjust the prices
        for i in range(1, len(back)):
            # If the instrument_id_data changes
            if (back["instrument_id_data"].iloc[i] != back["instrument_id_data"].iloc[i - 1]):
                adj_factor += back["close"].iloc[i - 1] - back["close"].iloc[i]
            # Adjust the prices
            # back["open_adj"].iloc[i] += adj_factor
            # back["high_adj"].iloc[i] += adj_factor
            # back["low_adj"].iloc[i] += adj_factor
            # back["close_adj"].iloc[i] += adj_factor
            back.at[i, "open_adj"] = back["open"].iloc[i] + adj_factor
            back.at[i, "high_adj"] = back["high"].iloc[i] + adj_factor
            back.at[i, "low_adj"] = back["low"].iloc[i] + adj_factor
            back.at[i, "close_adj"] = back["close"].iloc[i] + adj_factor
            # Store the adjustment factor
            # back["adj_factor"].iloc[i] = adj_factor
            back.at[i, "adj_factor"] = adj_factor

        # Flip the dataframe back
        self.backadjusted = back[::-1].copy()

    def format_carry(self) -> pd.DataFrame:
        # @NOTE: This method is used to format the carry data into the correct format for the database
        #  To find carry, we need to annualize the difference between the front and back month contracts.
        front = self.get_front().copy(deep=True)
        back = self.get_back().copy(deep=True)

        # Setting 'timestamp' as the index
        front.set_index("timestamp", inplace=True)
        back.set_index("timestamp", inplace=True)

        # Keep only OHLCV and expiration columns
        front = front[["open", "high", "low", "close", "volume", "expiration"]]
        back = back[["open", "high", "low", "close", "volume", "expiration"]]

        # Calculate the span of contracts in years... for example ES has a span of 3 months is 0.25 years
        # Some of the expirations are wrong, but we can find the difference between expirations in front using groupby

        # Rename the columns to front and back prefix and Capitalize the columns
        front.columns = [f"Front {col.capitalize()}" for col in front.columns]
        back.columns = [f"Back {col.capitalize()}" for col in back.columns]

        # Merge the front and back dataframes
        carry = pd.merge_asof(
            front.sort_index(),
            back.sort_index(),
            left_index=True,
            right_index=True,
            direction="nearest",
        )

        # Capitalize the index
        carry.index.name = carry.index.name.capitalize()

        # * Both volumes need to change from uint64 to int64 as Postgres does not support uint64
        # @NOTE: However, we need to cast the NaN values to previous values in ordre to convert to int64
        carry["Front Volume"] = carry["Front Volume"].ffill()
        carry["Back Volume"] = carry["Back Volume"].ffill()
        # Convert the volumes to int64
        carry = carry.astype({"Front Volume": np.int64, "Back Volume": np.int64})
        return carry

    def format_trend(self) -> pd.DataFrame:
        backad = self.get_backadjusted().copy(deep=True)
        # Setting 'timestamp' as the index
        backad.set_index("timestamp", inplace=True)
        # Keeping only necessary columns
        backad = backad[
            [
                "open_adj",
                "high_adj",
                "low_adj",
                "close_adj",
                "volume",
                "open",
                "high",
                "low",
                "close",
                "adj_factor",
                "raw_symbol",
                "expiration",
            ]
        ]

        # Rename columns to match the new format
        backad.columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Open Unadjusted",
            "High Unadjusted",
            "Low Unadjusted",
            "Close Unadjusted",
            "Adjustment Factor",
            "Contract Symbol",
            "Expiration",
        ]
        # Renaming the Index
        backad.index.name = "Timestamp"

        # Change the volume to int64 as Postgres does not support uint64
        # @NOTE: However, we need to cast the NaN values to previous values in ordre to convert to int64
        backad["Volume"] = backad["Volume"].ffill()
        # Convert the volumes to int64
        backad = backad.astype({"Volume": np.int64})

        return backad

    def store(self):
        # Store the backadjusted data
        self.format_trend().to_parquet(f"trend/{self.symbol}_trend.parquet")
        # Store the front month data and back month data
        self.format_carry().to_parquet(f"carry/{self.symbol}_carry.parquet")
        pass

    def trend(self, variance) -> pd.DataFrame:
        """
        The trend method is used to find the trend signal of the backadjusted data.
        The follwing steps are performed:
        1. Calculate the 2, 4, 8, 16, 32, 64, 128, 256 exponential moving averages of the close price.
        2. Find the crossovers of: 2-8, 4-16, 8-32, 16-64, 32-128, 64-256
        3. Find the risk adjusted forecasts using the standardDeviation class in price terms.
        4. Scale the crossovers by the absolute mean of all previous crossovers. Or use the LUT.
        5. Clip the scaled crossovers to -20, 20
        6. Concatenate the clipped and scaled crossovers and scale this final value by the Forecast Diversification Multiplier.
        7. Store the final value in the database.
        """
        # Get the backadjusted data
        data: pd.DataFrame = self.format_trend()
        trend: pd.DataFrame = (
            pd.DataFrame()
        )  # Create a new dataframe to store the trend data

        # Passinging Close and Unadjusted Close to the standard deviation class
        # stdDev = standardDeviation(data["Close"], data["Close Unadjusted"])

        trends = [2, 4, 8, 16, 32, 64, 128, 256]
        crossovers = [(2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256)]

        # Calculate the exponential moving averages crossovers and store them in the trend dataframe for t1, t2 in crossovers: trend[f"{t1}-{t2}"] = data["Close"].ewm(span=t1, min_periods=2).mean() - data["Close"].ewm(span=t2, min_periods=2).mean()
        for t1, t2 in crossovers:
            trend[f"{t1}-{t2}"] = (
                data["Close"].ewm(span=t1, min_periods=2).mean()
                - data["Close"].ewm(span=t2, min_periods=2).mean()
            )
        # Calculate the risk adjusted forecasts
        for t1, t2 in crossovers:
            trend[f"{t1}-{t2}"] /= ((variance ** 0.5) * data["Close Unadjusted"])

        # Scale the crossovers by the absolute mean of all previous crossovers
        # scalar_dict = {64: 1.91, 32: 2.79, 16: 4.1, 8: 5.95, 4: 8.53, 2: 12.1}
        scalar_dict = {}
        for t1, t2 in crossovers:
            scalar_dict[t1] = 10 / trend[f"{t1}-{t2}"].abs().mean()
        for t1, t2 in crossovers:
            trend[f"{t1}-{t2}"] = trend[f"{t1}-{t2}"] * scalar_dict[t1]

        # Clip the scaled crossovers to -20, 20
        for t1, t2 in crossovers:
            trend[f"{t1}-{t2}"] = trend[f"{t1}-{t2}"].clip(-20, 20)

        # Concatenate the clipped and scaled crossovers and apply the Forecast Diversification Multiplier
        # @WARN: Forecast Diversification Multiplier is not implemented and currently uses a LUT
        # FDM = {1: 1.0, 2: 1.03, 3: 1.08, 4: 1.13, 5: 1.19, 6: 1.26}
        """
        Calculating the Forecast Diversification Multiplier:

        Given N trading rule variations with an N x N correlation matrix of forecast values rho,
        and a vector of forecast weights w with length N and summing to 1: FDM = 1 / sqrt(w @ rho @ w.T)
        """
        trend["Forecast"] = 0.0
        # @NOTE: This can be adjusted to change the weights of the crossovers but we currently equal
        n = len(crossovers)
        weights = {64: 1 / n, 32: 1 / n, 16: 1 / n, 8: 1 / n, 4: 1 / n, 2: 1 / n}
        corr = trend[[f"{t1}-{t2}" for t1, t2 in crossovers]].corr()
        w = np.array([weights[t1] for t1, t2 in crossovers])
        fdm = 1 / np.sqrt(w @ corr @ w.T)
        for t1, t2 in crossovers:
            trend["Forecast"] += trend[f"{t1}-{t2}"] * weights[t1]
        trend["Forecast"] = trend["Forecast"] * fdm
        # CLip the final forecast to -20, 20
        trend["Forecast"] = trend["Forecast"].clip(-20, 20)

        return trend["Forecast"]

    def carry(self, variance):
        """
        The carry method is used to find the carry signal of the front and back month data.
        The following steps are performed:
        1. Calculate the difference between the front and back month contracts in price terms.
        2. Annualize the difference between the front and back month contracts.
        3. Find the Risk Adjusted Carry using the standardDeviation class in price terms.
        4. Smooth the Risk Adjusted Carry using an Exponential Moving Average.
        """
        data = self.format_carry()
        raw_carry = data["Front Close"] - data["Back Close"]
        # The raw carry is found by finding the difference between the price of the currently held contract and the price of the next contract
        annualized_carry = (raw_carry 
            / ((data["Back Expiration"] - data["Front Expiration"]).dt.days / 365).mean())
        
        # The annualized carry is found by dividing the raw carry by the number of days between the expiration of the front and back month contracts
        raw = self.format_trend()
        # ! Reaccess the functionality of the standard deviation class
        stdDev = standardDeviation(
            adjusted_price=raw["Close"], current_price=raw["Close Unadjusted"])

        risk_adjusted_carry = annualized_carry / ((variance ** 0.5) * data["Close Unadjusted"])

        spans = [5, 20, 60, 120]
        smoothed_carries = []
        for span in spans:
            smoothed_carries.append(
                risk_adjusted_carry.ewm(span=span, min_periods=1).mean()
            )
        smoothed_carries = pd.concat(smoothed_carries, axis=1)
        # Combine the smoothed carries into a single series
        smoothed_carries.columns = spans
        # Set the column names to the spans

        corr = smoothed_carries.corr()
        n = len(spans)
        weights = [1 / n] * n
        weights = np.array(weights)
        fdm = 1 / (weights @ corr.to_numpy() @ weights)
        combined = smoothed_carries.mean(axis=1) * fdm
        # Cap carry from -20 to 20
        combined = combined.clip(-20, 20)
        return combined


class Transforms:
    """
    The Transforms class is used as a collection of transformations to be applied to the raw data via a passed in list of symbols.

    Methods:
    * INITIALIZATION & GETTER METHODS
    - apply: Applies the transformations to all the symbols in the list
    - __init__: Initializes the class with a list of symbols
    - get_transformations: Returns the transformations applied to the symbols
    - get_symbols: Returns the symbols passed in to the class
    * STORAGE METHODS
    The storage methods are intended to store the transformed data the repository as well as the postgres database
    - store: Stores the transformed data in the repository
    - load: loads the transformed data to the postgres database
    """

    def __init__(self, symbols: list):
        # @param symbols: list of symbols to apply transformations to
        self.symbols = symbols
        self.transformations = []
        # * Applying the transformations to all passed in symbols
        self.apply()

    def apply(self):
        for symbol in self.symbols:
            t = Transformation(symbol)
            self.transformations.append(t)

    def get_transformations(self) -> list:
        return self.transformations

    def get_symbols(self) -> list:
        return self.symbols

    def store(self):
        for t in self.transformations:
            t.store()
        pass

    def trend(self) -> pd.DataFrame:
        """
        The trend method within the Transforms class combines the trend signals of all the symbols into a single dataframe.
        """
        merged = pd.DataFrame()
        for t in self.transformations:
            merged[t.get_symbol()] = t.trend()
        return merged

    def carry(self) -> pd.DataFrame:
        """
        The carry method within the Transforms class combines the carry signals of all the symbols into a single dataframe.
        """
        merged = pd.DataFrame()
        for t in self.transformations:
            merged[t.get_symbol()] = t.carry()
        return merged

    def get_risk(self) -> dict[str, standardDeviation]:
        """
        The get_risk method within the Transforms class returns the standard deviation of the backadjusted data for all the symbols.
        """
        risk_dct : dict[str, standardDeviation] = {t.get_symbol(): t.get_risk() for t in self.transformations}
        
        merged_df : pd.DataFrame = pd.DataFrame()
        for key, df in risk_dct.items():
            df.name = key
            if merged_df.empty:
                merged_df = df
            else:
                merged_df = pd.concat([merged_df, df], axis=1, join="outer")
        return merged_df

    def get_current_price(self) -> pd.DataFrame:
        """
        The get_current_price

        Args:
        - None

        Returns:
        - pd.DataFrame: The current price of the data for each of the symbols
        """
        price_dct = {t.get_symbol(): t.get_current_price() for t in self.transformations}

        merged_df : pd.DataFrame = pd.DataFrame()
        for key, df in price_dct.items():
            df.name = key
            if merged_df.empty:
                merged_df = df
            else:
                merged_df = pd.concat([merged_df, df], axis=1, join="outer")
        return merged_df 
    
    def get_open_interest(self) -> pd.DataFrame:
        """
        The get_open_interest method returns the open interest of the instruments in the portfolio

        Args:
        - None

        Returns:
        - A dictionary of the open interest of the instruments in the portfolio
        """
        return {t.get_symbol(): t.get_open_interest() for t in self.transformations}

    def load(self):
        # @NOTE: Using the Trend and Carry classes within the base.py file, we can store the data in the postgres database
        transforms_bar = tqdm.tqdm(self.transformations, desc="Storing Data")
        trends = []
        carries = []
        for t in transforms_bar:
            trend = Trend(data=t.format_trend(), symbol=t.get_symbol())
            carry = Carry(data=t.format_carry(), symbol=t.get_symbol())
            trends.append(trend)
            carries.append(carry)

        # Store the data in the postgres database
        trends_bar = tqdm.tqdm(trends, desc="Storing Trend Data")
        carries_bar = tqdm.tqdm(carries, desc="Storing Carry Data")
        for trend in trends_bar:
            trend.store()
        for carry in carries_bar:
            carry.store()

        # Store the combined trend and carry signals

    def signals(self, variances : pd.DataFrame) -> pd.DataFrame:
        trend_signals: dict[str, pd.DataFrame] = {
            t.get_symbol(): t.trend(variances[t.get_symbol()])
            for t in self.transformations
        }
        carry_signals: dict[str, pd.DataFrame] = {
            t.get_symbol(): t.carry(variances[t.get_symbol()])
            for t in self.transformations
        }
        # Combine the trend and carry signals but of a 60% weight to the trend signals and 40% weight to the carry signals
        combined_signals: dict[str, pd.Series] = {
            t.get_symbol(): trend_signals[t.get_symbol()]*0.6 + carry_signals[t.get_symbol()]*0.4 
            for t in self.transformations
        }

        # Turn the combined signals into a json
        combined_dataframe: pd.DataFrame = pd.DataFrame(combined_signals)
        return combined_dataframe

    def get_price_tables(self) -> dict[str, pd.DataFrame]:
        trend_tables = {}
        t: Transformation
        for t in self.transformations:
            trend_tables[t.get_symbol()] = t.format_trend()
        return trend_tables
