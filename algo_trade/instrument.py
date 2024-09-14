import os
from typing import Any, Dict, Tuple, Optional
from abc import ABC
import databento as db
import pandas as pd
from enum import Enum 
from dotenv import load_dotenv

from algo_trade.contract import ASSET, DATASET, CATALOG, Agg, RollType, Contract, ContractType

load_dotenv()

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

    def __init__(self, symbol: str, dataset: str, currency : str, exchange : str, instrument_type: Optional['InstrumentType'] = None, multiplier : float = 1.0, ib_symbol : str | None = None):
        self._symbol = symbol
        self._ib_symbol = ib_symbol if ib_symbol is not None else symbol
        self._dataset = dataset
        self.client: db.Historical = db.Historical(os.getenv("DATABENTO_API_KEY"))
        self.asset: ASSET
        self.multiplier = multiplier
        self._currency = currency
        self._exchange = exchange

        if instrument_type is not None:
            self.__class__ = instrument_type.value

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
    def dataset(self) -> str:
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
    Future class is a representation of a future instrument within the financial markets
    Within future contructs we can have multiple contracts that represent the same underlying asset so we need to be able to handle multiple contracts like a front month and back month contract
    To implement this we will have a list of contracts that the future instrument will handle

    Attributes:
    symbol: str - The symbol of the future instrument
    dataset: str - The dataset of the future instrument
    contracts: dict[str, Contract] - The contracts of the future instrument
    front: Contract - The front month contract of the future instrument
    back: Contract - The back month contract of the future instrument


    Methods:
    -   add_contract(contract: Contract, contract_type: ContractType) -> None - Adds a contract to the future instrument
    """

    def __init__(self, symbol: str, dataset: str, multiplier: float = 1.0):
        super().__init__(symbol, dataset)
        self.multiplier: float = multiplier
        self.contracts: dict[str, Contract] = {}
        self.asset = ASSET.FUT
        self._front: Contract
        self._back: Contract
        self._price: pd.Series

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
        self._front = value

    @front.deleter
    def front(self) -> None:
        """
        Deletes the front month contract of the future instrument

        Args:
        None

        Returns:
        None
        """
        del self._front

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
        if self._price.empty:
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
        self._price = value

    @price.deleter
    def price(self) -> None:
        """
        Deletes the price of the future instrument

        Args:
        None

        Returns:
        None
        """
        del self._price

    def __str__(self) -> str:
        return f"Future: {self.symbol} - {self.dataset}"

    def __repr__(self) -> str:
        return f"Future: {self.symbol} - {self.dataset}"

    def get_contracts(self) -> dict[str, Contract]:
        """
        Returns the contracts of the future instrument

        Args:
        None

        Returns:
        dict[str, Contract]: The contracts of the future instrument
        """
        if self.contracts == {}:
            raise ValueError("No Contracts are present")
        else:
            return self.contracts

    def get_front(self) -> Contract:
        return self.front

    def get_back(self) -> Contract:
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
        contract: Contract = Contract(
            instrument=self.symbol,
            dataset=DATASET.from_str(self.dataset),
            schema=schema,
        )

        if name is None:
            name = f"{contract.get_instrument()}-{roll_type}-{contract_type}"

        contract.construct(
            client=self.client, roll_type=roll_type, contract_type=contract_type
        )

        self.contracts[name] = contract
        if contract_type == ContractType.FRONT:
            self.front = contract
            self.price = contract.backadjusted
        elif contract_type == ContractType.BACK:
            self.back = contract

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
            dataset=DATASET.from_str(self.dataset),
            schema=Agg.DAILY,
            catalog=CATALOG.NORGATE,
        )

        if name is None:
            name = f"{contract.get_instrument()}"

        contract.construct_norgate()

        self.contracts[name] = contract
        self.front = contract
        self.price = contract.backadjusted

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
            #* For equation see: 
            # https://qoppac.blogspot.com/2023/02/percentage-or-price-differences-when.html
            self._percent_change : pd.Series = (
                self.price - self.price.shift(1)) / self.front.get_close().shift(1)

            self._percent_change.name = self.name

        return self._percent_change

class InstrumentType(Enum):
    FUTURE = Future

    @classmethod
    def from_str(cls, value: str) -> "InstrumentType":
        """
        Converts a string to a InstrumentType enum based on the value to the Enum name and not value
        so "FUTURE" -> FUTURE

        Args:
            - value: str - The value to convert to a InstrumentType enum

        Returns:
            - InstrumentType: The InstrumentType enum
        """
        try:
            return cls[value.upper()]
        except ValueError:

            for member in cls:
                if member.name.lower() == value.lower():
                    return member

            raise ValueError(f"{value} is not a valid {cls.__name__}")
    

def initialize_instruments(instrument_df : pd.DataFrame) -> list[Instrument]:
    return [
        Instrument(
            symbol=row.loc['dataSymbol'],
            dataset=row.loc['dataSet'],
            currency=row.loc['currency'],
            exchange=row.loc['exchange'],
            instrument_type=InstrumentType.from_str(row.loc['instrumentType']),
            multiplier=row.loc['multiplier'],
            ib_symbol=row.loc['ibSymbol']
        )
        for n, row in instrument_df.iterrows()
    ]

if __name__ == "__main__":
    # lst = initialize_instruments(pd.read_csv('data/contract.csv'))
    # Testing the Bar class

    # sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    ex: str = "CME"
    bucket: list[str] = ["ES", "NQ", "RTY", "YM", "ZN"]
    futures: list[Future] = []
    for sym in bucket:
        fut: Future = Future(symbol=sym, dataset=ex)
        fut.add_data(schema=Agg.DAILY, roll_type=RollType.CALENDAR, contract_type=ContractType.FRONT)
        futures.append(fut)
    
    print(futures)
