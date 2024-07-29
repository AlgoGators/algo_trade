"""
* The base.py file is intended to be used as a collection of classes that will be used to store the transformed data in the repository as well as the postgres
The different classes are as follows:
- Trend: The trend class is used to store the backadjusted data in the database defining the table schema for each contracts.
- Carry: The carry class is used to store the front month and back month data in the database defining the table schema for each contract.
"""
import os
from typing import Any, Dict

import toml

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
config_dir = os.path.join(base_dir, 'config')
config_path = os.path.join(config_dir, 'config.toml')

config_data: Dict[str, Any] = toml.load(config_path)

# Setup PostgreSQL database parameters from the configuration data for later connection.
DB_PARAMS: Dict[str, Any] = {
    'trend': config_data['database']['db_trend'],
    'carry': config_data['database']['db_carry'],
    'user': config_data['database']['user'],
    'password': config_data['database']['password'],
    'host': config_data['server']['ip'],
    'port': config_data['database']['port']
}

import pandas as pd
# The ORM used to store and load the data to the database is SQLAlchemy
from sqlalchemy import (Column, DateTime, Float, Integer, MetaData, String,
                        Table, create_engine)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# * Declaring the engine for the ORM
trend_engine = create_engine(f"postgresql+psycopg2://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['trend']}")
carry_engine = create_engine(f"postgresql+psycopg2://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['carry']}")

# * Declaring the session for the ORM
TrendSession = sessionmaker(bind=trend_engine)
CarrySession = sessionmaker(bind=carry_engine)

"""
The trend and carry data exist within there each respective databases. Each contract is stored in a table to its respective database. The tables are named after the contract symbol in the following format: {symbol}_data where symbol is the contract symbol
The schema for the tables remains the same for each contract within a particular database. However, the schema for the trend and carry data is different.

The schema for the trend data is as follows:
- Timestamp: <datetime> The timestamp of the data
- Open: <float> The open price of the contract
- High: <float> The high price of the contract
- Low: <float> The low price of the contract
- Close: <float> The close price of the contract
- Volume: <int> The volume of the contract
- Open Unadjusted: <float> The open unadjusted price of the contract
- High Unadjusted: <float> The high unadjusted price of the contract
- Low Unadjusted: <float> The low unadjusted price of the contract
- Close Unadjusted: <float> The close unadjusted price of the contract
- Adjustment Factor: <float> The adjustment factor of the contract
- Contract Symbol: <string> The contract symbol
- Expiration: <datetime> The expiration date of the contract

The schema for the carry data is as follows:
- Timestamp: <datetime> The timestamp of the data 
- Front Open: <float> The open price of the front month contract
- Front High: <float> The high price of the front month contract
- Front Low: <float> The low price of the front month contract
- Front Close: <float> The close price of the front month contract
- Front Volume: <int> The volume of the front month contract
- Front Expiration: <datetime> The expiration date of the front month contract
- Back Open: <float> The open price of the back month contract
- Back High: <float> The high price of the back month contract
- Back Low: <float> The low price of the back month contract
- Back Close: <float> The close price of the back month contract
- Back Volume: <int> The volume of the back month contract
"""

# * Declaring the trend class
class Trend:
    def __init__(self, data: pd.DataFrame, symbol: str):
        self.data = data
        self.symbol = symbol

    def store(self):
        session = TrendSession()
        data = self.data.copy()
        data.reset_index(inplace=True)
        data.to_sql(f"{self.symbol}_data", trend_engine, if_exists='replace', index=False)
        session.close()
    
    def load(self):
        session = TrendSession()
        data = pd.read_sql(f"SELECT * FROM {self.symbol}_data", trend_engine)
        session.close()
        return data
    
    def delete(self):
        session = TrendSession()
        session.execute(f"DROP TABLE IF EXISTS {self.symbol}_data")
        session.close()

# * Declaring the carry class
class Carry:
    def __init__(self, data: pd.DataFrame, symbol: str):
        self.data = data
        self.symbol = symbol

    def store(self):
        session = CarrySession()
        data = self.data.copy()
        data.reset_index(inplace=True)
        data.to_sql(f"{self.symbol}_data", carry_engine, if_exists='replace', index=False)
        session.close()
    
    def load(self):
        session = CarrySession()
        data = pd.read_sql(f"SELECT * FROM {self.symbol}_data", carry_engine)
        session.close()
        return data
    
    def delete(self):
        session = CarrySession()
        session.execute(f"DROP TABLE IF EXISTS {self.symbol}_data")
        session.close()