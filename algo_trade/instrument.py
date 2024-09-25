"""
This module defines classes and functions for handling financial instruments, 
specifically focusing on futures contracts. It includes the following components:
Classes:
- SecurityType: An enumeration for different types of securities.
- Instrument: A base class for financial instruments, providing attributes and methods 
    for handling common properties like symbol, dataset, currency, and exchange.
- Future: A subclass of Instrument, representing futures contracts. It includes additional 
    attributes and methods for handling multiple contracts, front and back month contracts, 
    and price data.
Functions:
- initialize_instruments: Initializes a list of Instrument objects from a pandas DataFrame.
- fetch_futures_data: Asynchronously fetches data for a list of Future objects, with rate 
    limiting to avoid excessive concurrent requests.
- main: An asynchronous main function that initializes a list of Future objects, fetches 
    their data, and prints their prices.
The module also utilizes the databento library for historical data fetching and the dotenv 
library for loading environment variables.
"""

import asyncio
from enum import Enum
import os
from typing import Tuple, Optional, Type, cast

from dotenv import load_dotenv
import databento as db
import pandas as pd # type: ignore

from algo_trade.contract import DATASET, CATALOG, Agg, RollType, Contract, ContractType

load_dotenv()

class SecurityType(Enum):
    """
    SecurityType enum to represent the different types of securities
    """

    FUTURE = ('Future', 'FUT')

    def __init__(self, obj_name : str, string : str):
        self._obj_name : str = obj_name
        self.string : str = string

    @property
    def obj(self) -> Type['Instrument']:
        """Dynamically resolve the object class when accessed"""
        if isinstance(self._obj_name, str):
            instrument_class = globals()[self._obj_name]
            if not issubclass(instrument_class, Instrument):
                raise ValueError(f"{self._obj_name} is not a valid Instrument class")
            return cast(Type[Instrument], instrument_class)
        return self._obj_name

    @classmethod
    def from_str(cls, value: str) -> "SecurityType":
        """
        Converts a string to a SecurityType enum based on the value to the Enum name and not value
        so "FUTURE" -> FUTURE

        Args:
            - value: str - The value to convert to a SecurityType enum

        Returns:
            - SecurityType: The SecurityType enum
        """
        try:
            return cls[value.upper()]
        except KeyError as exc:
            # If exact match fails, look for a case-insensitive match
            for member in cls:
                if member.name.lower() == value.lower():
                    return member

            # reraise the original exception if no match was found
            raise KeyError(f"{value} is not a valid {cls.__name__}") from exc

class Instrument():
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

    security_type: SecurityType

    def __init__(
            self,
            symbol: str,
            dataset: DATASET,
            currency : str,
            exchange : str,
            security_type: Optional['SecurityType'] = None,
            multiplier : float = 1.0,
            ib_symbol : str | None = None
        ) -> None:
        self._symbol = symbol
        self._ib_symbol = ib_symbol if ib_symbol is not None else symbol
        self._dataset = dataset
        self.client: db.Historical = db.Historical(os.getenv("DATABENTO_API_KEY"))
        self.multiplier = multiplier
        self._currency = currency
        self._exchange = exchange

        if security_type is not None:
            self.__class__ = security_type.obj

    @property
    def symbol(self) -> str:
        """
        Returns the symbol of the instrument

        Args:
        None

        Returns:
        str: The symbol of the instrument
        """
        return self._symbol

    @property
    def ib_symbol(self) -> str:
        """
        Returns the IBKR symbol of the instrument
        
        Args:
        None
        
        Returns:
        str: The IBKR symbol of the instrument
        """
        return self._ib_symbol

    @property
    def currency(self) -> str:
        """
        Returns the currency of the instrument

        Args:
        None

        Returns:
        str: The currency the instrument is denominated in
        """
        return self._currency

    @property
    def exchange(self) -> str:
        """
        Returns the exchange the instrument trades on

        Args:
        None

        Returns:
        str: The exchange the instrument trades on
        """
        return self._exchange

    @property
    def dataset(self) -> DATASET:
        """
        Returns the dataset of the instrument

        Args:
        None

        Returns:
        str: The dataset of the instrument
        """
        return self._dataset

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

    def get_collection(self) -> Tuple[str, str]:
        """
        Returns the symbol and dataset of the instrument

        Args:
        None

        Returns:
        Tuple[str, str]: The symbol and dataset of the instrument
        """
        return (self.symbol, self.dataset)

    #! PRICE MUST BE THE PRICE THAT YOU WANT TO USE FOR BACKTESTING
    @property
    def price(self) -> pd.Series:
        """
        Returns the prices of the instrument

        Args:
        None

        Returns:
        pd.Series: The prices of the instrument
        """
        raise NotImplementedError()

    @price.setter
    def price(self, value: pd.Series) -> None:
        """
        Sets the prices of the instrument

        Args:
        value: pd.Series - The prices of the instrument

        Returns:
        None
        """
        if not isinstance(value, pd.Series):
            raise ValueError("Price must be a pd.Series object")
        self._price = value

    @property
    def percent_returns(self) -> pd.Series:
        """
        Returns the percent returns of the instrument

        Args:
        None

        Returns:
        pd.Series: The percent returns of the instrument
        """
        raise NotImplementedError()

class Future(Instrument):
    """
    Future class is a representation of a future instrument
    Within future we can have multiple contracts representing the same underlying asset
    So we need to be able to handle multiple contracts like a front month and back month contract
    To implement this we will have a list of contracts that the future instrument will handle

    Attributes:
    symbol: str - The symbol of the future instrument
    dataset: str - The dataset of the future instrument
    contracts: dict[str, Contract] - The contracts of the future instrument
    front: Contract - The front month contract of the future instrument
    back: Contract - The back month contract of the future instrument
    """

    security_type = SecurityType.FUTURE
    def __init__(
            self,
            symbol : str,
            dataset : DATASET,
            currency : str,
            exchange : str,
            multiplier : float,
            ib_symbol : str | None = None
        ) -> None:

        super().__init__(
            symbol=symbol,
            dataset=dataset,
            currency=currency,
            exchange=exchange,
            security_type=SecurityType.FUTURE,
            multiplier=multiplier,
            ib_symbol=ib_symbol
        )

        # self.contracts: dict[str, Contract] = {}

    @property
    def front(self) -> Contract:
        """
        Returns the front month contract of the future instrument

        Args:
        None

        Returns:
        Bar: The front month contract of the future instrument
        """
        if not hasattr(self, "_front"):
            raise ValueError("Front is empty")

        return self._front

    @front.setter
    def front(self, value: Contract) -> None:
        """
        Sets the front month contract of the future instrument

        Args:
        value: Bar - The front month contract of the future instrument

        Returns:
        None
        """
        if not isinstance(value, Contract):
            raise ValueError("Front must be a Contract object")

        self._front = value

    @property
    def price(self) -> pd.Series:
        """
        Returns the price of the future instrument

        Args:
        None

        Returns:
        pd.Series: The price of the future instrument
        """
        if not hasattr(self, "_price"):
            raise ValueError("Price is empty")

        return self._price

    @price.setter
    def price(self, value: pd.Series) -> None:
        """
        Sets the price of the future instrument

        Args:
        value: pd.Series - The price of the future instrument

        Returns:
        None
        """
        if not isinstance(value, pd.Series):
            raise ValueError("Price must be a pd.Series object")

        self._price = value

    @property
    def back(self) -> Contract:
        """
        Returns the back month contract of the future instrument

        Args:
        None

        Returns:
        Bar: The back month contract of the future instrument
        """
        if not hasattr(self, "_back"):
            raise ValueError("Back is empty")

        return self._back

    @back.setter
    def back(self, value: Contract) -> None:
        """
        Sets the back month contract of the future instrument

        Args:
        value: Bar - The back month contract of the future instrument

        Returns:
        None
        """
        if not isinstance(value, Contract):
            raise ValueError("Back must be a Contract object")

        self._back = value

    @property
    def contracts(self) -> dict[str, Contract]:
        """
        Returns the contracts of the future instrument

        Args:
        None

        Returns:
        dict[str, Contract]: The contracts of the future instrument
        """
        if not hasattr(self, "_contracts"):
            self._contracts : dict[str, Contract] = {}

        return self._contracts
    
    @contracts.setter
    def contracts(self, value: dict[str, Contract]) -> None:
        """
        Sets the contracts of the future instrument

        Args:
        value: dict[str, Contract] - The contracts of the future instrument

        Returns:
        None
        """
        if not isinstance(value, dict):
            raise ValueError("Contracts must be a dict object")

        self._contracts = value

    def __str__(self) -> str:
        return f"Future: {self.symbol} - {self.dataset}"

    def __repr__(self) -> str:
        return f"Future: {self.symbol} - {self.dataset}"

    def add_norgate_data(self, name: Optional[str] = None) -> None:
        """
        Adds data to the future instrument but first creates a bar object based on the schema

        Args:
        name: Optional[str] - The name of the bar

        Returns:
        None
        """
        contract: Contract = Contract(
            instrument=self.symbol,
            dataset=self.dataset,
            schema=Agg.DAILY,
            catalog=CATALOG.NORGATE,
        )

        if name is None:
            name = f"{contract.get_instrument()}"

        contract.construct_norgate()

        self.contracts[name] = contract
        self.front = contract
        self.price = contract.backadjusted

    async def add_data_async(
        self,
        schema: Agg,
        roll_type: RollType,
        contract_type: ContractType,
        name: Optional[str] = None,
    ) -> None:
        """
        Asynchronously adds data to the future instrument,
        but first creates a bar object based on the schema

        Args:
        schema: Schema.BAR - The schema of the bar
        roll_type: RollType - The roll type of the bar
        contract_type: ContractType - The contract type of the bar
        name: Optional[str] - The name of the bar

        Returns:
        None
        """
        contract: Contract = Contract(
            instrument=self.symbol,
            dataset=self.dataset,
            schema=schema,
        )

        if name is None:
            name = f"{contract.get_instrument()}-{roll_type}-{contract_type}"
        # Add a sleep to the task to avoid rate limiting
        await asyncio.sleep(3)
        try:
            await contract.construct_async(
                client=self.client, roll_type=roll_type, contract_type=contract_type
            )
            if contract_type == ContractType.FRONT:
                self.front = contract
                self.price = contract.backadjusted
            elif contract_type == ContractType.BACK:
                self.back = contract

        except Exception as e:
            raise e

    @property
    def percent_returns(self) -> pd.Series:
        """
        Returns the percent returns of the future instrument

        Args:
        None

        Returns:
        pd.Series: The percent returns of the future instrument
        """

        if not hasattr(self, "_percent_change"):
            # * For equation see:
            # https://qoppac.blogspot.com/2023/02/percentage-or-price-differences-when.html
            self._percent_change = (
                self.price - self.price.shift(1)
            ) / self.front.get_close().shift(1)

            self._percent_change.name = self.name

        return self._percent_change

def initialize_instruments(instrument_df : pd.DataFrame) -> list[Instrument]:
    """
    Initializes a list of Instrument objects from a pandas DataFrame
    """
    return [
        Instrument(
            symbol=row.loc['dataSymbol'],
            dataset=DATASET.from_str(row.loc['dataSet']),
            currency=row.loc['currency'],
            exchange=row.loc['exchange'],
            security_type=SecurityType.from_str(row.loc['instrumentType']),
            multiplier=row.loc['multiplier'],
            ib_symbol=row.loc['ibSymbol']
        )
        for n, row in instrument_df.iterrows()
    ]

async def fetch_futures_data(futures : list[Future], rate: int = 5) -> None:
    """
    Fetches the data for the futures instruments asynchronously

    The fetch_futures_data function fetches the data for the futures instruments asynchronously 
    using asyncio and a semaphore to limit the number of concurrent requests.

    Args:
    futures: list[Future] - The list of future instruments to fetch data for
    rate: int - The rate limit for the number of concurrent requests

    Returns:
    None
    """
    semaphore = asyncio.Semaphore(rate)
    async def fetch_with_semaphore(future: Future) -> None:
        async with semaphore:
            await future.add_data_async(Agg.DAILY, RollType.CALENDAR, ContractType.FRONT)
    tasks = []
    for future in futures:
        task = asyncio.create_task(fetch_with_semaphore(future))
        tasks.append(task)

    await asyncio.gather(*tasks)

async def main() -> None:
    """
    Main function for the instrument module
    """
    ex: DATASET = DATASET.GLOBEX
    bucket: list[str] = ["ES", "NQ", "RTY", "YM", "ZN"]
    multipliers: dict[str, float] = {
        "ES": 50,
        "NQ": 20,
        "RTY": 50,
        "YM": 5,
        "ZN": 1000,
    }
    futures: list[Future] = []

    tasks = []

    for sym in bucket:
        fut: Future = Future(
            symbol=sym,
            dataset=ex,
            multiplier=multipliers[sym],
            currency="USD",
            exchange="GLOBEX"
        )

        task = asyncio.create_task(
            fut.add_data_async(Agg.DAILY, RollType.CALENDAR, ContractType.FRONT))
        tasks.append(task)
        futures.append(fut)

    await asyncio.gather(*tasks)

    print("Futures:")

    for fut in futures:
        print(fut.price)

if __name__ == "__main__":
    asyncio.run(main())
