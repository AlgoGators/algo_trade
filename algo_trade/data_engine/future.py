import asyncio
import os
from typing import Any, Dict, Tuple, List, Literal, Optional
from abc import abstractmethod, ABC
from enum import StrEnum, Enum
from pathlib import Path
from dataclasses import dataclass

import aiohttp
import databento as db
import pandas as pd
import toml

# Building out class structure to backadjust the futures data
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
config_dir = os.path.join(base_dir, "config")
config_path = os.path.join(config_dir, "config.toml")

config: Dict[str, Any] = toml.load(config_path)


# TODO: Add more vendor catalogs such as Norgate
class CATALOG(StrEnum):
    DATABENTO = f"data/catalog/databento/"

class ASSET(StrEnum):
    FUT = "FUT"
    OPT = "OPT"
    EQ = "EQ"

# TODO: Add more datasets
class DATASET(StrEnum):
    CME = "GLBX.MDP3"


# TODO: Add more schemas
class Agg(StrEnum):
    DAILY = "ohlcv-1d"
    HOURLY = "ohlcv-1h"
    MINUTE = "ohlcv-1m"
    SECOND = "ohlcv-1s"
    def __str__(self):
        return self.value
    
    def __repr__(self):
        return self.value

class RollType(StrEnum):
    CALENDAR = "c"
    OPEN_INTEREST = "n"
    VOLUME = "v"
    def __str__(self):
        return self.value
    
    def __repr__(self):
        return self.value

class ContractType(StrEnum):
    FRONT = "0"
    BACK = "1"
    def __str__(self):
        return self.value
    
    def __repr__(self):
        return self.value


class Bar():
    """
    Bar class to act as a base class for all bar classes
    
    Attributes:
    -   instrument_id: str - The instrument_id of the bar
    -   schema: Schema.BAR - The schema of the bar
    -   catalog: CATALOG - The catalog location of existing instrument data
    -   data: db.DBNStore - The data of the bar
    -   definitions: db.DBNStore - The definitions of the bar
    -   timestamp: pd.Timestamp - The timestamp of the bar
    -   open: pd.Series - The open price of the bar
    -   high: pd.Series - The high price of the bar
    -   low: pd.Series - The low price of the bar
    -   close: pd.Series - The close price of the bar
    -   volume: pd.Series - The volume of the bar

    Methods:

    GETTERS:
    -   get_timestamp() -> pd.Series - Returns the timestamp of the bar as a series
    -   get_open() -> pd.Series - Returns the open price of the bar as a series
    -   get_high() -> pd.Series - Returns the high price of the bar as a series
    -   get_low() -> pd.Series - Returns the low price of the bar as a series
    -   get_close() -> pd.Series - Returns the close price of the bar as a series
    -   get_volume() -> pd.Series - Returns the volume of the bar as a series
    -   get_bar() -> pd.DataFrame - Returns the bar as a dataframe
    -   get_backadjusted() -> pd.DataFrame - Returns the backadjusted bar as a dataframe

    CONSTRUCTORS:
    -   construct() -> None - Constructs the bar by first attempting to retrieve the data and definitions from the data catalog
    """
    def __init__(self, instrument_id: str, dataset: DATASET, schema: Agg, catalog: CATALOG = CATALOG.DATABENTO):
        self.data: pd.DataFrame
        self.definitions: pd.DataFrame
        self.timestamp: pd.Series
        self.open: pd.Series
        self.high: pd.Series
        self.low: pd.Series
        self.close: pd.Series
        self.volume: pd.Series
        self.instrument_id: str = instrument_id
        self.dataset: DATASET = dataset
        self.schema: Agg = schema
        self.catalog: CATALOG = catalog

    def __str__(self) -> str:
        return f"Bar: {self.instrument_id} - {self.dataset} - {self.schema}"

    def __repr__(self) -> str:
        return f"Bar: {self.instrument_id} - {self.dataset} - {self.schema}"

    def get_instrument_id(self) -> str:
        """
        Returns the instrument_id of the bar
        
        Args:
        None
        
        Returns:
        str: The instrument_id of the bar
        """
        return self.instrument_id

    def get_dataset(self) -> DATASET:
        """
        Returns the dataset of the bar
        
        Args:
        None
        
        Returns:
        DATASET: The dataset of the bar
        """
        return self.dataset

    def get_schema(self) -> Agg:
        """
        Returns the schema of the bar
        
        Args:
        None
        
        Returns:
        Schema.BAR: The schema of the bar
        """
        return self.schema

    def get_catalog(self) -> CATALOG:
        """
        Returns the catalog location of the existing instrument data
        
        Args:
        None
        
        Returns:
        CATALOG: The catalog location of the existing instrument data
        """
        return self.catalog

    def get_timestamp(self) -> pd.Series:
        """
        Returns the timestamp of the bar as a series
        
        Args:
        None
        
        Returns:
        pd.Series: The timestamp of the bar as a series
        """
        return self.timestamp
    
    def get_open(self) -> pd.Series:
        """
        Returns the open price of the bar as a series
        
        Args:
        None
        
        Returns:
        pd.Series: The open price of the bar as a series
        """
        return self.open
    
    def get_high(self) -> pd.Series:
        """
        Returns the high price of the bar as a series
        
        Args:
        None
        
        Returns:
        pd.Series: The high price of the bar as a series
        """
        return self.high
    
    def get_low(self) -> pd.Series:
        """
        Returns the low price of the bar as a series
        
        Args:
        None
        
        Returns:
        pd.Series: The low price of the bar as a series
        """
        return self.low
    
    def get_close(self) -> pd.Series:
        """
        Returns the close price of the bar as a series
        
        Args:
        None
        
        Returns:
        pd.Series: The close price of the bar as a series
        """
        return self.close
    
    def get_volume(self) -> pd.Series:
        """
        Returns the volume of the bar as a series
        
        Args:
        None
        
        Returns:
        pd.Series: The volume of the bar as a series
        """
        return self.volume
    
    def get_bar(self) -> pd.DataFrame:
        """
        Returns the bar as a dataframe
        
        Args:
        None
        
        Returns:
        pd.DataFrame: The bar as a dataframe
        """
        return pd.DataFrame({
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume
        })
    
    def get_backadjusted(self) -> pd.DataFrame:
        """
        Returns the backadjusted bar as a dataframe
        
        Args:
        None
        
        Returns:
        pd.DataFrame: The backadjusted bar as a dataframe
        """
        # TODO: Implement backadjustment
        if self.definitions or self.data is None:
            raise ValueError("Data and Definitions are not present")

        # Backadjust the bar
        pass

    def construct(self, client: db.Historical, roll_type: RollType, contract_type: ContractType, range: dict[str, str]) -> None:
        """
        Constructs the bar by first attempting to retrieve the data and definitions from the data catalog
        
        Args:
        None
        
        Returns:
        None
        """
        # TODO: Implement construct method

        data_path: Path = Path(f"{self.catalog}/{self.instrument_id}/{self.schema}/{roll_type}-{contract_type}-data.parquet")
        definitions_path: Path = Path(f"{self.catalog}/{self.instrument_id}/{self.schema}/{roll_type}-{contract_type}-definitions.parquet")

        if data_path.exists() and definitions_path.exists():
            try:
                self.data = pd.read_parquet(data_path)
                self.definitions = pd.read_parquet(definitions_path)
            except Exception as e:
                print(f"Error: {e}")
                return

            start: pd.Timestamp = pd.Timestamp(range[self.dataset]["start"])
            end: pd.Timestamp = pd.Timestamp(range[self.dataset]["end"])

            data_end: pd.Timestamp = self.data["Timestamp"].iloc[-1]
            definitions_end: pd.Timestamp = self.definitions["Timestamp"].iloc[-1]

            # Check if the data and definitions are up to date

            if data_end != end or definitions_end != end:
                print("Data and Definitions are not up to date for {self.instrument_id}")
                symbols: str = f"{self.instrument_id}.{roll_type}.{contract_type}"
                new_data: db.DBNStore = client.timeseries.get_range(dataset=self.dataset, symbols=symbols, schema=db.Schema.from_str(self.schema), start=data_end, end=end)
                new_definitions: db.DBNStore = new_data.request_full_definitions(client=client)

                # Combine new data with existing data and skip duplicates if they exist
                self.data = pd.concat([self.data, new_data.to_df()]).drop_duplicates(subset=["Timestamp"], keep="last")
                self.definitions = pd.concat([self.definitions, new_definitions.to_df()]).drop_duplicates(subset=["Timestamp"], keep="last")

                # Save the new data and definitions to the catalog
                self.data.to_parquet(data_path)
                self.definitions.to_parquet(definitions_path)

                # Set the timestamp, open, high, low, close, and volume
                self.timestamp = self.data["Timestamp"]
                self.open = self.data["Open"]
                self.high = self.data["High"]
                self.low = self.data["Low"]
                self.close = self.data["Close"]
                self.volume = self.data["Volume"]

        else:
            print(f"Data and Definitions not present for {self.instrument_id}")
            # Submit a job request to retrieve the data and definitions
            symbols: str = f"{self.instrument_id}.{roll_type}.{contract_type}"
            # TODO: Implement job request submission
            # details: dict[str, Any] = client.batch.submit_job(dataset=self.dataset, symbols=symbols, schema=db.Schema.from_str(self.schema), encoding=db.Encoding.DBN start=start, end=end, stype_in=db.SType.CONTINUOUS, split_duration=db.SplitDuration.NONE)
            # print(f"Job Request Submitted: {details["symbols"]} - {details["schema"]} - {details["start"]} - {details["end"]}")
            data: db.DBNStore = client.timeseries.get_range(dataset=self.dataset, symbols=symbols, schema=db.Schema.from_str(self.schema), start=start, end=end)
            definitions: db.DBNStore = data.request_full_definitions(client=client)

            # Save the data and definitions to the catalog
            data.to_parquet(data_path)
            definitions.to_parquet(definitions_path)

            self.data = data.to_df()
            self.definitions = definitions.to_df()

            # Set the timestamp, open, high, low, close, and volume
            self.timestamp = self.data["Timestamp"]
            self.open = self.data["Open"]
            self.high = self.data["High"]
            self.low = self.data["Low"]
            self.close = self.data["Close"]
            self.volume = self.data["Volume"]

            # WARNING: The API "should" be able to handle data requests under 5 GB but have had issues in the pass with large requests
            return

class Instrument(ABC):
    """
    Instrument class to act as a base class for all asset classes

    Attributes:
    symbol: str - The symbol of the instrument
    dataset: str - The dataset of the instrument

    Methods:
    get_symbol() -> str - Returns the symbol of the instrument
    get_dataset() -> str - Returns the dataset of the instrument
    get_collection() -> Tuple[str, str] - Returns the symbol and dataset of the instrument

    The instrument class is an 

    """
    def __init__(self, symbol: str, dataset: str):
        self.symbol = symbol
        self.dataset = dataset
        self.client: db.Historical = db.Historical(config["databento"]["api_historical"])
        self.asset: Optional[ASSET] = None
        

    def get_symbol(self) -> str:
        """
        Returns the symbol of the instrument

        Args:
        None

        Returns:
        str: The symbol of the instrument
        """
        return self.symbol

    def get_dataset(self) -> str:
        """
        Returns the dataset of the instrument

        Args:
        None

        Returns:
        str: The dataset of the instrument
        """
        return self.dataset
    
    def get_collection(self) -> Tuple[str, str]:
        """
        Returns the symbol and dataset of the instrument

        Args:
        None

        Returns:
        Tuple[str, str]: The symbol and dataset of the instrument
        """
        return (self.symbol, self.dataset)

class Future(Instrument):
    """
    Future class is a representation of a future instrument within the financial markets
    Within future contructs we can have multiple contracts that represent the same underlying asset so we need to be able to handle multiple contracts like a front month and back month contract
    To implement this we will have a list of contracts that the future instrument will handle

    Attributes:
    symbol: str - The symbol of the future instrument
    dataset: str - The dataset of the future instrument
    contracts: List[str] - The list of contracts that the future instrument will handle

    Methods:
    -   add_data(data: Data) -> None - Adds data to the future instrument
    """

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
