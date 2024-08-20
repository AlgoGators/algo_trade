import os
from typing import Any, Dict, Tuple, Optional
from abc import ABC
from enum import StrEnum 
from pathlib import Path

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

    @classmethod
    def from_str(cls, value: str) -> "DATASET":
        """
        Converts a string to a DATASET enum based on the value to the Enum name and not value
        so "CME" -> DATASET.CME

        Args:
            - value: str - The value to convert to a DATASET enum

        Returns:
            - DATASET: The DATASET enum
        """
        try:
            return cls[value.upper()]
        except ValueError:

            for member in cls:
                if member.name.lower() == value.lower():
                    return member

            raise ValueError(f"{value} is not a valid {cls.__name__}")


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


class Bar:
    """
    Bar class to act as a base class for all bar classes

    Attributes:
    -   instrument: str - The instrument_id of the bar
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
    -   expiration: pd.Series - The expiration of the bar
    -   instrument_id: pd.Series - The instrument_id of the bar

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
    -   get_expiration() -> pd.Series - Returns the expiration of the bar as a series
    -   get_instrument_id() -> pd.Series - Returns the instrument_id of the bar as a series

    CONSTRUCTORS:
    -   construct() -> None - Constructs the bar by first attempting to retrieve the data and definitions from the data catalog
    """

    # TODO: Refactor to use the @property decorator for getters and setters

    def __init__(
        self,
        instrument: str,
        dataset: DATASET,
        schema: Agg,
        catalog: CATALOG = CATALOG.DATABENTO,
    ):
        self.data: pd.DataFrame
        self.definitions: pd.DataFrame
        self.timestamp: pd.Index
        self.open: pd.Series
        self.high: pd.Series
        self.low: pd.Series
        self.close: pd.Series
        self.volume: pd.Series
        self.expiration: pd.Series
        self.instrument_id: pd.Series
        self.backadjusted: pd.Series = pd.Series()
        self.instrument: str = instrument
        self.dataset: DATASET = dataset
        self.schema: Agg = schema
        self.catalog: CATALOG = catalog

    def __str__(self) -> str:
        return f"Bar: {self.instrument} - {self.dataset} - {self.schema}"

    def __repr__(self) -> str:
        return f"Bar: {self.instrument} - {self.dataset} - {self.schema}"

    def get_instrument(self) -> str:
        """
        Returns the instrument_id of the bar

        Args:
        None

        Returns:
        str: The instrument_id of the bar
        """
        return self.instrument

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

    def get_timestamp(self) -> pd.Index:
        """
        Returns the timestamp of the bar as a series

        Args:
        None

        Returns:
        pd.Index: The timestamp of the bar as a pandas index
        """
        if self.timestamp.empty:
            raise ValueError("Timestamp is empty")
        return self.timestamp

    def get_open(self) -> pd.Series:
        """
        Returns the open price of the bar as a series

        Args:
        None

        Returns:
        pd.Series: The open price of the bar as a series
        """
        if self.open.empty:
            raise ValueError("Open is empty")
        return self.open

    def get_high(self) -> pd.Series:
        """
        Returns the high price of the bar as a series

        Args:
        None

        Returns:
        pd.Series: The high price of the bar as a series
        """
        if self.high.empty:
            raise ValueError("High is empty")
        return self.high

    def get_low(self) -> pd.Series:
        """
        Returns the low price of the bar as a series

        Args:
        None

        Returns:
        pd.Series: The low price of the bar as a series
        """
        if self.low.empty:
            raise ValueError("Low is empty")
        return self.low

    def get_close(self) -> pd.Series:
        """
        Returns the close price of the bar as a series

        Args:
        None

        Returns:
        pd.Series: The close price of the bar as a series
        """
        if self.close.empty:
            raise ValueError("Close is empty")
        return self.close

    def get_volume(self) -> pd.Series:
        """
        Returns the volume of the bar as a series

        Args:
        None

        Returns:
        pd.Series: The volume of the bar as a series
        """
        if self.volume.empty:
            raise ValueError("Volume is empty")
        return self.volume

    def get_expiration(self) -> pd.Series:
        """
        Returns the expiration of the bar as a series

        Args:
        None

        Returns:
        pd.Series: The expiration of the bar as a series
        """
        if self.expiration.empty:
            raise ValueError("Expiration is empty")
        return self.expiration

    def get_instrument_id(self) -> pd.Series:
        """
        Returns the instrument_id of the bar as a series

        Args:
        None

        Returns:
        pd.Series: The instrument_id of the bar as a series
        """
        if self.instrument_id.empty:
            raise ValueError("Instrument ID is empty")
        return self.instrument_id

    def get_bar(self) -> pd.DataFrame:
        """
        Returns the bar as a dataframe

        Args:
        None

        Returns:
        pd.DataFrame: The bar as a dataframe
        """
        return pd.DataFrame(
            {
                "timestamp": self.get_timestamp(),
                "open": self.get_open(),
                "high": self.get_high(),
                "low": self.get_low(),
                "close": self.get_close(),
                "volume": self.get_volume(),
            }
        )

    def get_backadjusted(self) -> pd.Series:
        """
        Returns the backadjusted series as a pandas series

        BACKADJUSTMENT:
        Backadjustment is the process of removing sudden gaps in pricing data when a contracts rolls over from the front to the back.
        While it does not preserve the integrity of the data, it gives a better representation of true pricemovements.
        The process goes as follows:
            1. Iterate backwards through a frame of both pricing data, and expiration(or another alias) data.
            2. When a contract switches to the previous contract, calculate the difference between the current contract and previous contract.
            3. Apply that previous adjustment to the current contract and all other following contracts keeping a rolling value
            4. Repeat!

        This method of futures adjustment is know as the panama canal method

        Args:
        None

        Returns:
        pd.DataFrame: The backadjusted bar as a dataframe
        """
        if self.definitions.empty or self.data.empty:
            raise ValueError("Data and Definitions are not present")
        elif self.open.empty or self.high.empty or self.low.empty or self.close.empty or self.volume.empty:
            raise ValueError("Open, High, Low, Close, or Volume is empty")
        elif self.backadjusted.empty:
            # Perform backadjustment on close prices and return the backadjusted series
            # TODO: Check logic for backadjustment
            backadjusted: pd.DataFrame = self.data.copy()[["close", "instrument_id"]]
            backadjusted.sort_index(ascending=False, inplace=True)
            cumm_adj: float = 0.0
            adj: float = 0.0
            for i in range(1, len(backadjusted)):
                if backadjusted.iloc[i]["instrument_id"] != backadjusted.iloc[i - 1]["instrument_id"]:
                    adj = backadjusted["close"].iloc[i - 1] - backadjusted["close"].iloc[i]
                    cumm_adj += adj
                    # Adjust all following prices with slicing
                    backadjusted.loc[backadjusted.index[i]:, "close"] += adj

            backadjusted.sort_index(ascending=True, inplace=True)
            self.backadjusted = pd.Series(backadjusted["close"])
            return self.backadjusted
        else:
            return self.backadjusted

    def _set_exp(self, data: pd.DataFrame, definitions: pd.DataFrame) -> pd.Series:
        """
        This _set_exp method is used to create an the correct timeseries that follows the the daily data of the OHLCV data versus the sparse data of the defintions which contains the correct expiration

        Args:
            - instrument_ids: pd.Series - The daily individual instrument_id
            - definitions: pd.Series - The definitions of our instruments which include both our expirations as well as matching instrument_id to our data series.
        """

        # Assert all instrument_ids in data are within definitions
        assert data["instrument_id"].isin(definitions["instrument_id"]).all(bool_only=True)

        # We need to index our definitons by the instrument_id
        exp_df: pd.DataFrame = definitions.reset_index()[["expiration", "instrument_id"]].set_index("instrument_id").drop_duplicates()

        # We then need to map our instrument_ids to the correct expiration date using the definitions while preserving the data frame index
        data["expiration"] = data["instrument_id"].map(exp_df["expiration"])
        # Finally we need to set the index of our instrument ids to the same index as our data using the timestamp
        expirations: pd.Series = data["expiration"]
        return expirations


    def construct(
        self, client: db.Historical, roll_type: RollType, contract_type: ContractType
    ) -> None:
        """
        Constructs the bar by first attempting to retrieve the data and definitions from the data catalog

        Args:
        -   client: db.Historical - The client to use to retrieve the data and definitions
        -   roll_type: RollType - The roll type of the bar
        -   contract_type: ContractType - The contract type of the bar

        Returns:
        None
        """
        # TODO: Implement construct method

        data_path: Path = Path(
            f"{self.catalog}/{self.instrument}/{self.schema}/{roll_type}-{contract_type}-data.parquet"
        )
        definitions_path: Path = Path(
            f"{self.catalog}/{self.instrument}/{self.schema}/{roll_type}-{contract_type}-definitions.parquet"
        )

        range: dict[str, str] = client.metadata.get_dataset_range(dataset=self.dataset)
        start: pd.Timestamp = pd.Timestamp(range["start"])
        end: pd.Timestamp = pd.Timestamp(range["end"])

        if data_path.exists() and definitions_path.exists():
            try:
                self.data = pd.read_parquet(data_path)
                self.definitions = pd.read_parquet(definitions_path)
            except Exception as e:
                print(f"Error: {e}")
                return

            data_end: pd.Timestamp = pd.Timestamp(self.data.index[-1])
            definitions_end: pd.Timestamp = pd.Timestamp(self.definitions.index[-1])

            # Check if the data and definitions are up to date

            if data_end != end or definitions_end != end:
                print(
                    "Data and Definitions are not up to date for {self.instrument_id}"
                )
                # Try to retrieve the new data and definitions but if failed then do not update
                try:
                    symbols: str = f"{self.instrument}.{roll_type}.{contract_type}"
                    new_data: db.DBNStore = client.timeseries.get_range(
                        dataset=self.dataset,
                        symbols=[symbols],
                        schema=db.Schema.from_str(self.schema),
                        start=data_end,
                        end=end,
                        stype_in=db.SType.CONTINUOUS,
                        stype_out=db.SType.INSTRUMENT_ID,
                    )
                    new_definitions: db.DBNStore = new_data.request_full_definitions(
                        client=client
                    )
                    # Combine new data with existing data and skip duplicates if they exist based on index
                    self.data = pd.concat([self.data, new_data.to_df()]).drop_duplicates()
                    self.definitions = pd.concat([self.definitions, new_definitions.to_df()]).drop_duplicates()
                except Exception as e:
                    print(f"Error: {e}")

            # Save the new data and definitions to the catalog
            self.data.to_parquet(data_path)
            self.definitions.to_parquet(definitions_path)

            # Set the timestamp, open, high, low, close, and volume
            self.timestamp = self.data.index
            self.open = pd.Series(self.data["open"])
            self.high = pd.Series(self.data["high"])
            self.low = pd.Series(self.data["low"])
            self.close = pd.Series(self.data["close"])
            self.volume = pd.Series(self.data["volume"])
            self.instrument_id = pd.Series(self.definitions["instrument_id"])
            self.expiration = self._set_exp(self.data.copy(), self.definitions.copy())

        else:
            print(f"Data and Definitions not present for {self.instrument}")
            # Submit a job request to retrieve the data and definitions
            symbols: str = f"{self.instrument}.{roll_type}.{contract_type}"
            # TODO: Implement job request submission
            # details: dict[str, Any] = client.batch.submit_job(dataset=self.dataset, symbols=symbols, schema=db.Schema.from_str(self.schema), encoding=db.Encoding.DBN start=start, end=end, stype_in=db.SType.CONTINUOUS, split_duration=db.SplitDuration.NONE)
            # print(f"Job Request Submitted: {details["symbols"]} - {details["schema"]} - {details["start"]} - {details["end"]}")
            data: db.DBNStore = client.timeseries.get_range(
                dataset=str(self.dataset),
                symbols=[symbols],
                schema=db.Schema.from_str(self.schema),
                start=start,
                end=end,
                stype_in=db.SType.CONTINUOUS,
                stype_out=db.SType.INSTRUMENT_ID,
            )
            definitions: db.DBNStore = data.request_full_definitions(client=client)

            # Make the directories if they do not exist
            data_path.parent.mkdir(parents=True, exist_ok=True)
            definitions_path.parent.mkdir(parents=True, exist_ok=True)
            # Save the data and definitions to the catalog
            data.to_parquet(data_path)
            definitions.to_parquet(definitions_path)

            self.data = data.to_df()
            self.definitions = definitions.to_df()

            # Set the timestamp, open, high, low, close, and volume
            self.timestamp = self.data.index
            self.open = pd.Series(self.data["open"])
            self.high = pd.Series(self.data["high"])
            self.low = pd.Series(self.data["low"])
            self.close = pd.Series(self.data["close"])
            self.volume = pd.Series(self.data["volume"])
            self.instrument_id = pd.Series(self.definitions["instrument_id"])
            self.expiration = self._set_exp(self.data.copy(), self.definitions.copy())

            # WARN: The API "should" be able to handle data requests under 5 GB but have had issues in the pass with large requests
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
        self.client: db.Historical = db.Historical(
            config["databento"]["api_historical"]
        )
        self.asset: ASSET

    def get_symbol(self) -> str:
        """
        Returns the symbol of the instrument

        Args:
        None

        Returns:
        str: The symbol of the instrument
        """
        return self.symbol

    @property
    def name(self) -> str:
        """
        Returns the name of the instrument

        Args:
        None

        Returns:
        str: The name of the instrument
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

    def get_asset(self) -> ASSET:
        """
        Returns the asset of the instrument

        Args:
        None

        Returns:
        str: The asset of the instrument
        """
        return self.asset

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
    data: dict[str, Data] - The data of the future instrument


    Methods:
    -   add_bar(bar: Bar, roll_type: RollType, contract_type: ContractType, name: Optional[str] = None) -> None - Adds data to the future instrument
    """

    def __init__(self, symbol: str, dataset: str, multiplier: float = 1.0):
        super().__init__(symbol, dataset)
        self.multiplier: float = multiplier
        self.bars: dict[str, Bar] = {}
        self.asset = ASSET.FUT
        self.__front__: Bar
        self.__back__: Bar
        self.__price__: pd.Series


    @property
    def front(self) -> Bar:
        """
        Returns the front month contract of the future instrument

        Args:
        None

        Returns:
        Bar: The front month contract of the future instrument
        """
        if self.__front__ is None:
            raise ValueError("Front is empty")
        return self.__front__

    @front.setter
    def front(self, value: Bar) -> None:
        """
        Sets the front month contract of the future instrument

        Args:
        value: Bar - The front month contract of the future instrument

        Returns:
        None
        """
        self.__front__ = value

    @front.deleter
    def front(self) -> None:
        """
        Deletes the front month contract of the future instrument

        Args:
        None

        Returns:
        None
        """
        del self.__front__

    front.__doc__ = """
    The front month contract of the future instrument

    Args:
        
    Returns:
        pd.Series: The front month contract of the future instrument
    """

    @property
    def price(self) -> pd.Series:
        """
        Returns the price of the future instrument

        Args:
        None

        Returns:
        pd.Series: The price of the future instrument
        """
        if self.__price__.empty:
            raise ValueError("Price is empty")
        return self.__price__

    @price.setter
    def price(self, value: pd.Series) -> None:
        """
        Sets the price of the future instrument

        Args:
        value: pd.Series - The price of the future instrument

        Returns:
        None
        """
        self.__price__ = value

    @price.deleter
    def price(self) -> None:
        """
        Deletes the price of the future instrument

        Args:
        None

        Returns:
        None
        """
        del self.__price__
    
    def __str__(self) -> str:
        return f"Future: {self.symbol} - {self.dataset}"

    def __repr__(self) -> str:
        return f"Future: {self.symbol} - {self.dataset}"

    def get_bars(self) -> dict[str, Bar]:
        """
        Returns the bars of the future instrument

        Args:
        None

        Returns:
        dict[str, Bar]: The bars of the future instrument
        """
        if self.bars == {}:
            raise ValueError("Bars are empty")
        else:
            return self.bars

    def get_front(self) -> Bar:
        """
        Returns the front month contract of the future instrument

        Args:
        None

        Returns:
        Bar: The front month contract of the future instrument
        """
        if self.front is None:
            raise ValueError("Front is empty")
        else:
            return self.front

    def get_back(self) -> Bar:
        """
        Returns the back month contract of the future instrument

        Args:
        None

        Returns:
        Bar: The back month contract of the future instrument
        """
        if self.back is None:
            raise ValueError("Back is empty")
        else:
            return self.back


    def add_data(
        self,
        schema: Agg,
        roll_type: RollType,
        contract_type: ContractType,
        name: Optional[str] = None,
    ) -> None:
        """
        Adds data to the future instrument but first creates a bar object based on the schema

        Args:
        schema: Schema.BAR - The schema of the bar
        roll_type: RollType - The roll type of the bar
        contract_type: ContractType - The contract type of the bar
        name: Optional[str] - The name of the bar

        Returns:
        None
        """
        bar: Bar = Bar(instrument=self.symbol, dataset=DATASET.from_str(self.dataset), schema=schema)

        if name is None:
            name = f"{bar.get_instrument()}-{roll_type}-{contract_type}"

        bar.construct(client=self.client, roll_type=roll_type, contract_type=contract_type)

        self.bars[name] = bar
        if contract_type == ContractType.FRONT:
            self.front = bar
            self.price = bar.get_close() * self.multiplier
        elif contract_type == ContractType.BACK:
            self.back = bar


if __name__ == "__main__":
    # Testing the Bar class
    # Set sys.path to the base directory
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    future: Future = Future("ES", "CME")
    future.add_data(Agg.DAILY, RollType.CALENDAR, ContractType.FRONT)
    exp = future.get_front().expiration

