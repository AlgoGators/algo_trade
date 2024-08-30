import os
from typing import Any, Dict, Tuple, Optional
from abc import ABC
import databento as db
import pandas as pd
import toml
from enum import Enum

from algo_trade.contract import ASSET, DATASET, Agg, RollType, Contract, ContractType

# Building out class structure to backadjust the futures data
base_dir = os.path.dirname(os.path.dirname(__file__))
config_dir = os.path.join(base_dir, "config")
config_path = os.path.join(config_dir, "config.toml")

config: Dict[str, Any] = toml.load(config_path)

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

    def __init__(self, symbol: str, dataset: str, instrument_type: Optional['InstrumentType'] = None):
        self._symbol = symbol
        self._dataset = dataset
        self.client: db.Historical = db.Historical(
            config["databento"]["api_historical"]
        )
        self.asset: ASSET

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
            self.price = contract.get_backadjusted()
        elif contract_type == ContractType.BACK:
            self.back = contract

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
    return [Instrument(row.loc['dataSymbol'], row.loc['dataSet'], InstrumentType.from_str(row.loc['instrumentType'])) for n, row in instrument_df.iterrows()]

if __name__ == "__main__":
    lst = initialize_instruments(pd.read_csv('data/contract.csv'))
    # Testing the Bar class
    quit()

    # sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    future: Future = Future("ES", "CME")
    future.add_data(Agg.DAILY, RollType.CALENDAR, ContractType.FRONT)
    exp = future.get_front().expiration
