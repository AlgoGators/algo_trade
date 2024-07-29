import asyncio
import os
from typing import Any, Dict

import aiohttp
import databento as db
import pandas as pd
import toml

# Building out class structure to backadjust the futures data
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
config_dir = os.path.join(base_dir, "config")
config_path = os.path.join(config_dir, "config.toml")

config: Dict[str, Any] = toml.load(config_path)


class Future:
    def __init__(self, symbol: str, dataset: str):
        # Loop through dataframe and create a new datafram with backadjusted prices
        self.symbol = symbol
        self.dataset = dataset
        # Open Config File

    # Getters
    def get_symbol(self) -> str:
        return self.symbol

    def get_dataset(self) -> str:
        return self.dataset

    def get_collection(self):
        return (self.symbol, self.dataset)


class Historical(Future):
    def __init__(
        self,
        symbol: str,
        dataset: str,
        start_date: str,
        end_date: str,
        wait: bool = False,
    ):
        super().__init__(symbol, dataset)

        self.start = start_date
        self.end = end_date
        self.client = db.Historical(config["databento"]["api_historical"])
        self.set_coro()
        # If wait is True, then wait on the build function as a higher entity will collect all the coroutines and run them
        if not wait:
            asyncio.run(self.build())

    def set_coro(self):
        # Retrieve futures data from databento
        # Define the number of rolls to be used and roll methodolgy
        # c = calender front month, n = open interest, v = volume
        rolls_rule = ["c.0", "c.1"]
        contract_with_rolls = [f"{self.symbol}.{roll}" for roll in rolls_rule]
        self.data_coro = self.client.timeseries.get_range_async(
            dataset=self.dataset,
            symbols=contract_with_rolls,
            schema="OHLCV-1d",
            start=self.start,
            end=self.end,
            stype_in="continuous",
        )
        self.definitions_coro = self.client.timeseries.get_range_async(
            dataset=self.dataset,
            symbols=contract_with_rolls,
            schema="definition",
            start=self.start,
            end=self.end,
            stype_in="continuous",
        )

    # Run the coroutines
    async def build(self):
        # Timeouts for the coroutines... 5 minutes
        timeout = 60 * 5
        try:
            # Run the coroutines
            data, definitions = await asyncio.gather(
                asyncio.wait_for(self.data_coro, timeout),
                asyncio.wait_for(self.definitions_coro, timeout),
            )
        except asyncio.TimeoutError:
            print(f"Timeout Error: Data not retrieved for {self.symbol}")
            return
        except aiohttp.ClientPayloadError as e:
            print(f"Error: {e} for {self.symbol}")
            return

        # Store the data and definitions
        self.data = data
        self.definitions = definitions
        # Build the raw dataframe
        self.build_raw()
        # Store the raw dataframe
        self.store()

    # Build the raw dataframe
    def build_raw(self):
        # Inner merges the data and definitions
        data_df = self.data.to_df()
        definitions_df = self.definitions.to_df()
        # self.raw: pd.DataFrame = pd.merge(
        #     data_df,
        #     definitions_df,
        #     how="inner",
        #     left_index=True,
        #     right_on=definitions_df.index,
        #     suffixes=("_data", "_definition"),
        # )
        data_df.index.name = "timestamp"
        definitions_df.index.name = "timestamp"
        self.raw = pd.merge_asof(
            data_df,
            definitions_df,
            on="timestamp",
            direction="nearest",
            suffixes=("_data", "_definition"),
        )

    def store(self):
        # Save the raw dataframe as a parquet file in raw directory
        try:
            self.raw.to_parquet(f"tmp/{self.symbol}_{self.start}_{self.end}.parquet")
        except AttributeError:
            print(f"Error: {self.symbol} has no raw data... failed to store")

    # Getters
    def get_coro(self) -> list:
        # Return the list of futures
        return [self.data_coro, self.definitions_coro]

    def get_contract_id(self):
        return self.data.to_df()["instrument_id"].iloc[-1]

    def get_raw(self) -> pd.DataFrame:
        return self.raw

    def get_definitions(self):
        return self.definitions.to_df()


class Live(Future):
    def __init__(self, symbol: str, dataset: str, live_date: str):
        super().__init__(symbol, dataset)

        with open("config.toml", "rb") as file:
            config = tl.load(file)

        # Store live date
        self.live_date = live_date

        # Using Live API key
        self.client = db.Historical(config["databento"]["api_historical"])

        self.get_data()

    # Get Live Data from Databento
    def get_data(self):
        # Get raw data from Databento
        rolls_rule = ["c.0"]
        contract_with_rolls = [f"{self.symbol}.{roll}" for roll in rolls_rule]
        self.data = self.client.timeseries.get_range(
            dataset=self.dataset,
            symbols=contract_with_rolls,
            schema="OHLCV-1d",
            start=self.live_date,
            stype_in="continuous",
        )
        # Storing in data attribute
        return

    def get_definition(self, date: str):
        data = self.client.timeseries.get_range(
            dataset=self.dataset,
            stype_in="continuous",
            symbols=f"{self.symbol}.c.0",
            schema="definition",
            start=self.live_date,
        ).to_df()
        return data

    def get_raw(self) -> pd.DataFrame:
        return self.data.to_df()
        # THIS MIGHT CHANGE BASED ON THE LIVE API
