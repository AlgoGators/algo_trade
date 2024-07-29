# SQL Manager for the database
import os
from typing import Any, Dict

import pandas as pd
import sqlalchemy as sa
import sqlalchemy.ext.declarative as dec
import toml
import tqdm
from future import Future, Historical, Live
from portfolio import HistoricalPortfolio, Portfolio

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
config_dir = os.path.join(base_dir, 'config')
config_path = os.path.join(config_dir, 'config.toml')

config: Dict[str, Any] = toml.load(config_path)

class Manager:
    def __init__(self):
        # ! CHANGE BACK TO TREND
        self.db_params : dict[str, any] = {
            'db_trend': config['database']['demo'],
            'db_carry': config['database']['db_carry'],
            'user': config['database']['user'],
            'password': config['database']['password'],
            'host': config['server']['ip'],
            'port': config['database']['port']
        }
        # Create the engine
        trend_uri = f"postgresql+psycopg2://{self.db_params['user']}:{self.db_params['password']}@{self.db_params['host']}:{self.db_params['port']}/{self.db_params['db_trend']}"
        carry_uri = f"postgresql+psycopg2://{self.db_params['user']}:{self.db_params['password']}@{self.db_params['host']}:{self.db_params['port']}/{self.db_params['db_carry']}"
        # Establish a connection
        self.trend_engine = sa.create_engine(trend_uri)
        self.carry_engine = sa.create_engine(carry_uri)

        # self.trend_session = sa.orm.sessionmaker(bind=self.trend_engine)
        # self.carry_session = sa.orm.sessionmaker(bind=self.carry_engine)
    
    # Testing the connection
    def test_connection(self):
        # Test connection to both trend and carry databases
        try:
            self.trend_engine.connect()
            self.carry_engine.connect()
            print("Connection successful")
            return True
        except:
            print("Connection failed")
            return False

    def del_trend_table(self, symbol: str):
        confirm = input(f"Are you sure you want to delete {symbol}_data? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Deletion aborted")
            return
        with self.trend_engine.connect() as conn:
            conn.execute(f"DELETE FROM {symbol}_data")
    
    def del_all_trend_tables(self):
        confirm = input(f"Are you sure you want to delete all tables? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Deletion aborted")
            return
        with self.trend_engine.connect() as conn:
            conn.execute("DELETE FROM *")

    def clear_carry_table(self, symbol: str):
        with self.carry_engine.connect() as conn:
            conn.execute(f"DELETE FROM {symbol}_data")
    
    def clear_all_carry_tables(self):
        with self.carry_engine.connect() as conn:
            conn.execute("DELETE FROM *")
    
    def get_trend_table(self, symbol: str) -> pd.DataFrame:
        with self.trend_engine.connect() as conn:
            return pd.read_sql(f"SELECT * FROM \"{symbol}_data\";", conn)
    
    def get_all_trend_tables(self) -> dict:
        with self.trend_engine.connect() as conn:
            return {table: pd.read_sql(f"SELECT * FROM {table}", conn) for table in self.trend_engine.table_names()}
    
    def get_carry_table(self, symbol: str) -> pd.DataFrame:
        with self.carry_engine.connect() as conn:
            return pd.read_sql(f"SELECT * FROM \"{symbol}_data\"", conn)
        
    def get_all_carry_tables(self) -> dict:
        with self.carry_engine.connect() as conn:
            return {table: pd.read_sql(f"SELECT * FROM {table}", conn) for table in self.carry_engine.table_names()}

    def create_trend_table(self, symbol: str):
        sa.Table(f"{symbol}_data", sa.MetaData(),
                    sa.Column('Date', sa.DateTime, primary_key=True),
                    sa.Column('Open', sa.Float),
                    sa.Column('High', sa.Float),
                    sa.Column('Low', sa.Float),
                    sa.Column('Close', sa.Float),
                    sa.Column('Volume', sa.Integer),
                    sa.Column('Backadjusted', sa.Float)).create(self.trend_engine)

    def fetch_existing_dates(self, symbol: str):
        session = self.trend_session()
        existing_dates_query = session.query(sa.column('Date')).from_table(sa.table(f"{symbol}_data"))
        existing_dates = [result.Date for result in existing_dates_query.all()]
        session.close()
        return existing_dates 
    # TODO: Create a function to create a carry table

class HistoricalManager(Manager):
    def __init__(self):
        super().__init__()
        
    # insert an entire portfolio into the db
    def insert_portfolio(self, portfolio: HistoricalPortfolio):
        # Using tqdm to show progress
        for symbol, future in tqdm.tqdm(portfolio.get_all_contracts().items()):
            self.insert_historical_trend(symbol=symbol, contract=future)
        
    # insert historical data into specific table
    def insert_historical_trend(self, symbol: str, contract: Historical):
        # Get the data
        data = contract.get_clean()
        # Check for a table if no table exists, create one
        inspect = sa.inspect(self.trend_engine)
        if not inspect.has_table(f"{symbol}_data"):
            print(f"Creating table {symbol}_data")
            self.create_trend_table(symbol)
            # Insert the data into the database
            for col in data.select_dtypes(include=['uint64']).columns:
                data[col] = data[col].astype('int64')
            with self.trend_engine.connect() as conn:
                data.to_sql(f"{symbol}_data", conn, if_exists='replace')
        else:
            print(f"Table {symbol}_data already exists")
            for col in data.select_dtypes(include=['uint64']).columns:
                data[col] = data[col].astype('int64')
            # Insert the data into the database
            with self.trend_engine.connect() as conn:
                data.to_sql(f"{symbol}_data", conn, if_exists='replace', index=False)
    
    # Check for existing dates in the database
    def update_historical_trend(self, symbol: str, contract: Historical):
        data = contract.get_clean()
        existing_dates = self.fetch_existing_dates(symbol)

        # Filter out the existing dates
        filtered_data = data[~data['Date'].isin(existing_dates)]

        # Insert the data into the database
    
class LiveManager(Manager):
    def __init__(self):
        super().__init__()

    def append_portfolio(self, portfolio: Portfolio):
        for symbol, future in portfolio.get_all_contracts().items():
            self.append_live_trend(symbol=symbol, contract=future)
    def insert_live_trend(self, symbol: str, contract: Live):
        # Get the data
        data = contract.get_clean()
        # Check for a table if no table exists, create one
        inspect = sa.inspect(self.trend_engine)
        if not inspect.has_table(f"{symbol}_data"):
            print(f"Creating table {symbol}_data")
            self.create_trend_table(symbol)
            # Insert the data into the database
            for col in data.select_dtypes(include=['uint64']).columns:
                data[col] = data[col].astype('int64')
            with self.trend_engine.connect() as conn:
                data.to_sql(f"{symbol}_data", conn, if_exists='replace')
        else:
            print(f"Table {symbol}_data already exists")
            for col in data.select_dtypes(include=['uint64']).columns:
                data[col] = data[col].astype('int64')
            # Insert the data into the database
            with self.trend_engine.connect() as conn:
                data.to_sql(f"{symbol}_data", conn, if_exists='replace', index=False)
