import os
from typing import Any, Dict

import pandas as pd
import psycopg2
import toml
from psycopg2 import sql
from psycopg2.extensions import connection as PGConnection
from risk_measures import RiskMeasures
from sqlalchemy import create_engine, text
from std_daily_price import standardDeviation

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
config_dir = os.path.join(base_dir, 'config')
config_path = os.path.join(config_dir, 'config.toml')

config_data: Dict[str, Any] = toml.load(config_path)

# Setup PostgreSQL database parameters from the configuration data for later connection.
DB_PARAMS: Dict[str, Any] = {
    "dbname": config_data["database"]["db_ng_trend"],
    "user": config_data["database"]["user"],
    "password": config_data["database"]["password"],
    "host": config_data["server"]["ip"],
    "port": config_data["database"]["port"],
}

carry_data_type_list = [
    "next-expiry-price",
    "front-month-price",
    "raw-carry",
    "annualized-raw-carry",
    "annualized-std-dev-price",
    "risk-adjusted-annual-carry",
    "smooth-ewma-5",
    "smooth-ewma-20",
    "smooth-ewma-60",
    "smooth-ewma-120",
    "scaled-ewma-5",
    "scaled-ewma-20",
    "scaled-ewma-60",
    "scaled-ewma-120",
    "capped-ewma-5",
    "capped-ewma-20",
    "capped-ewma-60",
    "capped-ewma-120",
    "raw-combined-forecast",
    "scaled-combined-forecast",
    "expiry-difference",
    "capped-combined-forecast",
]

trend_data_type_list = [
    "ewma-2",
    "ewma-4",
    "ewma-8",
    "ewma-16",
    "ewma-32",
    "ewma-64",
    "ewma-128",
    "ewma-256",
    "annualized-std-dev-price",
    "raw-2x8-forecast",
    "raw-4x16-forecast",
    "raw-8x32-forecast",
    "raw-16x64-forecast",
    "raw-32x128-forecast",
    "raw-64x256-forecast",
    "scaled-2x8-forecast",
    "scaled-4x16-forecast",
    "scaled-8x32-forecast",
    "scaled-16x64-forecast",
    "scaled-32x128-forecast",
    "scaled-64x256-forecast",
    "capped-2x8-forecast",
    "capped-4x16-forecast",
    "capped-8x32-forecast",
    "capped-16x64-forecast",
    "capped-32x128-forecast",
    "capped-64x256-forecast",
    "raw-combined-forecast",
    "scaled-combined-forecast",
    "capped-combined-forecast",
]

contracts_Norton = [
    "6A",
    "6B",
    "6C",
    "6E",
    "6J",
    "6M",
    "6N",
    "6S",
    "CT",
    "ES",
    "FCE",
    "FDAX",
    "FSMI",
    "GC",
    "GF",
    "HE",
    "HG",
    "KC",
    "KE",
    "LE",
    "LRC",
    "LSU",
    "NQ",
    "PL",
    "RB",
    "RTY",
    "SB",
    "SCN",
    "SI",
    "UB",
    "VX",
    "WBS",
    "YM",
    "ZC",
    "ZL",
    "ZM",
    "ZN",
    "ZR",
    "ZS",
    "ZW",
]

contracts_databento = [
    "6A",
    "6B",
    "6C",
    "6E",
    "6J",
    "6M",
    "6N",
    "6S",
    "CL",
    "ES",
    "GC",
    "GF",
    "HE",
    "HG",
    "LE",
    "MNQ",
    "MSF",
    "PL",
    "RB",
    "RTY",
    "SI",
    "UB",
    "YM",
    "ZC",
    "ZL",
    "ZM",
    "ZN",
    "ZR",
    "ZS",
    "ZW",
]


def fetch_symbol_dict() -> Dict[str, str]:
    # Define the path to the CSV file that contains mapping of symbols to datasets.
    path_to_csv: str = r"contracts.csv"
    # Load the CSV file into a pandas DataFrame.
    df: pd.DataFrame = pd.read_csv(path_to_csv)
    # Convert the DataFrame into a dictionary with 'Data Symbol' as keys and 'Data Set' as values for easy lookup.
    symbol_dict: Dict[str, str] = df.set_index("Data Symbol")["Data Set"].to_dict()

    return symbol_dict


def get_connection() -> PGConnection:
    # Establish and return a connection object to the PostgreSQL database using psycopg2.
    conn: PGConnection = psycopg2.connect(**DB_PARAMS)

    return conn


# function to create a table for a particular data type - the table's 1st column should be the date and then the
# rest of the columns should be the contract names from fetch_symbol_dict
def create_table_for_data_type(data_type: str) -> None:
    # Fetch the dictionary mapping symbols to their datasets.
    symbol_dict: Dict[str, str] = fetch_symbol_dict()

    # Get a database connection.
    conn: PGConnection = get_connection()

    # Open a new database cursor for executing SQL commands.
    with conn.cursor() as cur:
        # Starting with a 'date' column
        columns = ["date DATE"]

        # Add the rest of the columns. These should be contract names from the fetch_symbol_dict function.
        columns += [f'"{contract}" TEXT' for contract in symbol_dict.keys()]

        # Create table query
        create_table_query = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
            sql.Identifier(data_type), sql.SQL(", ").join(map(sql.SQL, columns))
        )

        # Execute the create table query
        cur.execute(create_table_query)

        # Commit the changes
        conn.commit()

        # Close the cursor.
        cur.close()

    # Close the connection.
    conn.close()


# connect to the database and pull the data
def get_data_from_db(dt) -> pd.DataFrame:
    # Get a database connection.
    conn: PGConnection = get_connection()
    # Fetch the dictionary mapping symbols to their datasets.
    symbol_dict: Dict[str, str] = fetch_symbol_dict()
    # Open a new database cursor for executing SQL commands.
    with conn.cursor() as cur:
        # SQL query template for creating a new table with necessary columns for financial data.
        create_table_query: str = f"""
        SELECT * FROM \"{dt}\";
        """
        # Execute the SQL query to create a new table for the symbol.
        cur.execute(create_table_query)
        # Fetch all the rows from the result of the executed query.

        columns = [col[0] for col in cur.description]

        rows = cur.fetchall()
        # Convert the rows into a pandas DataFrame.
        df = pd.DataFrame(rows, columns=columns)
        # Close the cursor.
        cur.close()
        # Close the connection.
        conn.close()

    return df


# DATE REFORMATTING FUNCTION
# This function helps to handle / and - formatted dates
def date_format_correction(dateIndex):
    newDateFormat = []
    for a, date in dateIndex.items():
        newFormat = ""
        if "/" in date:
            month = date[0:2]
            day = date[3:5]
            year = date[6:10]
            newFormat = year + "-" + month + "-" + day
            newDateFormat.append(newFormat)
        else:
            return dateIndex

    return newDateFormat


# DATA COLLECTION
# this function grabs the data from the postgres server for the specified database
# Parameters: contractParam -> list of strings for the contract names
#             db -> database that should be queried for the data collection
def getBothDict(contractParam, db):
    trendDict = getTrendDict(contractParam, db)
    carryDict = getCarryDict(contractParam, db)

    return trendDict, carryDict


def getTrendDict(contractParam, db):
    if db == "Norton":
        DB_PARAMS["dbname"] = "trenddata"
    elif db == "Bento":
        DB_PARAMS["dbname"] = "trend_databento"

    trendDict = {}
    for contract in contractParam:
        dt = contract + "_data"
        trendDict[contract] = get_data_from_db(dt)

    print("Trend Collected")
    return trendDict


def getCarryDict(contractParam, db):
    if db == "Norton":
        DB_PARAMS["dbname"] = "carrydata"
    elif db == "Bento":
        DB_PARAMS["dbname"] = "carry_databento"
    carryDict = {}
    for contract in contractParam:
        if db == "Norton":
            dt = contract + "_Data_Carry"
        elif db == "Bento":
            dt = contract + "_data"

        carryDict[contract] = get_data_from_db(dt)

    print("Carry Collected")
    return carryDict


# CARRY DATA CLASS
# takes in a parameter for carry dictionary data and the trend dictionary data from getBothDict
class CarryData:
    # number of ewma
    rule_count = 4
    FDM_DICT = {1: 1.0, 2: 1.02, 3: 1.03, 4: 1.04}
    fdm = FDM_DICT[rule_count]

    # initializes all of the different individual pd.DataFrames for Carry
    def __init__(self, carryDict, trendDict):
        self.front_month_price = CarryData.getFront(self, carryDict)
        self.next_expiry_price = CarryData.getNext(self, carryDict)
        self.raw_carry = self.front_month_price - self.next_expiry_price
        self.expiry_difference = CarryData.getExpiry(self, carryDict)
        self.annualized_raw_carry = self.raw_carry / self.expiry_difference
        self.annualized_std_dev_price = CarryData.getStdevCarry(trendDict)
        self.risk_adjusted_annual_carry = (
            self.annualized_raw_carry / self.annualized_std_dev_price
        )
        self.smooth_ewma_5 = CarryData.calculate_ewm(
            self, self.risk_adjusted_annual_carry, 5
        )
        self.smooth_ewma_20 = CarryData.calculate_ewm(
            self, self.risk_adjusted_annual_carry, 20
        )
        self.smooth_ewma_60 = CarryData.calculate_ewm(
            self, self.risk_adjusted_annual_carry, 60
        )
        self.smooth_ewma_120 = CarryData.calculate_ewm(
            self, self.risk_adjusted_annual_carry, 120
        )

        self.scaled_ewma_5 = self.smooth_ewma_5 * 30
        self.scaled_ewma_20 = self.smooth_ewma_20 * 30
        self.scaled_ewma_60 = self.smooth_ewma_60 * 30
        self.scaled_ewma_120 = self.smooth_ewma_120 * 30
        self.capped_ewma_5 = self.scaled_ewma_5.clip(-20, 20)
        self.capped_ewma_20 = self.scaled_ewma_20.clip(-20, 20)
        self.capped_ewma_60 = self.scaled_ewma_60.clip(-20, 20)
        self.capped_ewma_120 = self.scaled_ewma_120.clip(-20, 20)
        divisor = CarryData.rule_count
        self.raw_combined_forecast = (
            (self.capped_ewma_5 * 1 / divisor)
            + (self.capped_ewma_20 * 1 / divisor)
            + (self.capped_ewma_60 * 1 / divisor)
            + (self.capped_ewma_120 * 1 / divisor)
        )
        self.scaled_combined_forecast = self.raw_combined_forecast * CarryData.fdm
        self.capped_combined_forecast = self.scaled_combined_forecast.clip(-20, 20)
        self.variableDict = CarryData.getVariableDict(self)

        print("Carry Class Initialized")

    # forms the dictionary for all of the attributes for Carry Data
    def getVariableDict(self):
        dict = vars(self)
        return dict

    # creates connection to specified postgres database, should match the one used in data collection
    # converts dataframe to sql datatable, drops if exists and then replaces
    def updateCarry(self, database):
        DB_PARAMS["dbname"] = database

        connection_str = f'postgresql://{DB_PARAMS["user"]}:{DB_PARAMS["password"]}@{DB_PARAMS["host"]}:{DB_PARAMS["port"]}/{DB_PARAMS["dbname"]}'

        engine = create_engine(connection_str)
        with engine.connect() as conn:
            for name, df in self.variableDict.items():
                if type(df) == pd.DataFrame:
                    basic_historical_append(df, name, conn)

            conn.close()

    # gets the dataframe for front month data
    def getFront(self, carryDict):
        df = pd.DataFrame()
        for key, value in carryDict.items():
            data_dt = value
            index = data_dt.Date
            index = date_format_correction(index)
            price = data_dt.Price
            price.name = key
            price = price.set_axis(index)
            df = pd.concat([df, price], axis=1)

        df = df.sort_index(axis=0)

        return df

    # calculates the individual ewm for each contract after removing null dates
    def calculate_ewm(self, df, spanParam):
        dfNew = pd.DataFrame()
        for column in df:
            series = df[column]
            filteredSeries = series.dropna()
            filteredSeries = filteredSeries.ewm(span=spanParam).mean()
            filteredSeries.name = column
            dfNew = pd.concat([dfNew, filteredSeries], axis=1)

        dfNew = dfNew.sort_index(axis=0)

        return dfNew

    # gets the dataframe for the next expiry price
    def getNext(self, carryDict):
        df = pd.DataFrame()
        for key, value in carryDict.items():
            data_dt = value
            index = data_dt.Date
            index = date_format_correction(index)
            carry = data_dt.Carry
            carry.name = key
            carry = carry.set_axis(index)
            df = pd.concat([df, carry], axis=1)

        df = df.sort_index(axis=0)

        return df

    # gets the dataframe for the next expiration date
    def getExpiry(self, carryDict):
        df = pd.DataFrame()
        for key, value in carryDict.items():
            data_dt = value
            index = data_dt.Date
            index = date_format_correction(index)
            price_date = data_dt.Price_Contract
            price_date = price_date.set_axis(index)
            carry_date = data_dt.Carry_Contract
            carry_date = carry_date.set_axis(index)

            price_year_frac = CarryData._total_year_frac_from_contract_series(
                price_date
            )
            carry_year_frac = CarryData._total_year_frac_from_contract_series(
                carry_date
            )
            expiry_difference = carry_year_frac - price_year_frac
            expiry_difference.name = key

            df = pd.concat([df, expiry_difference], axis=1)

        df = df.sort_index(axis=0)

        return df

    # calculates year fraction
    def _total_year_frac_from_contract_series(x):
        years = CarryData._year_from_contract_series(x)
        month_frac = CarryData._month_as_year_frac_from_contract_series(x)

        return years + month_frac

    # Returns the year of a specific contract
    def _year_from_contract_series(x):
        return x // 100

    # Returns the month as a fraction of a year
    def _month_as_year_frac_from_contract_series(x):
        return CarryData._month_from_contract_series(x) / 12.0

    # Returns the month for a specific contract
    def _month_from_contract_series(x):
        return x % 100

    # gets the std dev of price annually using trend data
    def getStdevCarry(trendDict):
        df = pd.DataFrame()
        for key, value in trendDict.items():
            data_dt = value
            index = data_dt.Date
            index = date_format_correction(index)
            close = data_dt.Close
            unadj_close = data_dt.Unadj_Close
            close = close.set_axis(index)
            unadj_close = unadj_close.set_axis(index)

            stdDev = standardDeviation(close, unadj_close)
            stdev_annual = stdDev.annual_risk_price_terms()

            stdev_annual.name = key

            df = pd.concat([df, stdev_annual], axis=1)

        df = df.sort_index(axis=0)

        return df


# TREND CARRY CLASS
# calculates values for all the trend attributes using one parameter: trendDict
class TrendData:
    rule_count = 6
    scalar_dict = {64: 1.91, 32: 2.79, 16: 4.1, 8: 5.95, 4: 8.53, 2: 12.1}
    FDM_DICT = {1: 1.0, 2: 1.03, 3: 1.08, 4: 1.13, 5: 1.19, 6: 1.26}
    fdm = FDM_DICT[rule_count]

    # initialises all trend class attributes
    def __init__(self, trendDict):
        close = TrendData.getClose(self, trendDict)
        self.ewma_2 = TrendData.calculate_ewm(self, close, 2)
        self.ewma_4 = TrendData.calculate_ewm(self, close, 4)
        self.ewma_8 = TrendData.calculate_ewm(self, close, 8)
        self.ewma_16 = TrendData.calculate_ewm(self, close, 16)
        self.ewma_32 = TrendData.calculate_ewm(self, close, 32)
        self.ewma_64 = TrendData.calculate_ewm(self, close, 64)
        self.ewma_128 = TrendData.calculate_ewm(self, close, 128)
        self.ewma_256 = TrendData.calculate_ewm(self, close, 256)

        self.annualized_std_dev_price = TrendData.getStdevTrend(self, trendDict)
        self.raw_2x8_forecast = (
            self.ewma_2 - self.ewma_8
        ) / self.annualized_std_dev_price
        self.raw_4x16_forecast = (
            self.ewma_4 - self.ewma_16
        ) / self.annualized_std_dev_price
        self.raw_8x32_forecast = (
            self.ewma_8 - self.ewma_32
        ) / self.annualized_std_dev_price
        self.raw_16x64_forecast = (
            self.ewma_16 - self.ewma_64
        ) / self.annualized_std_dev_price
        self.raw_32x128_forecast = (
            self.ewma_32 - self.ewma_128
        ) / self.annualized_std_dev_price
        self.raw_64x256_forecast = (
            self.ewma_64 - self.ewma_256
        ) / self.annualized_std_dev_price
        self.scaled_2x8_forecast = self.raw_2x8_forecast * TrendData.scalar_dict[2]
        self.scaled_4x16_forecast = self.raw_4x16_forecast * TrendData.scalar_dict[4]
        self.scaled_8x32_forecast = self.raw_8x32_forecast * TrendData.scalar_dict[8]
        self.scaled_16x64_forecast = self.raw_16x64_forecast * TrendData.scalar_dict[16]
        self.scaled_32x128_forecast = (
            self.raw_32x128_forecast * TrendData.scalar_dict[32]
        )
        self.scaled_64x256_forecast = (
            self.raw_64x256_forecast * TrendData.scalar_dict[64]
        )
        self.capped_2x8_forecast = self.scaled_2x8_forecast.clip(-20, 20)
        self.capped_4x16_forecast = self.scaled_4x16_forecast.clip(-20, 20)
        self.capped_8x32_forecast = self.scaled_8x32_forecast.clip(-20, 20)
        self.capped_16x64_forecast = self.scaled_16x64_forecast.clip(-20, 20)
        self.capped_32x128_forecast = self.scaled_32x128_forecast.clip(-20, 20)
        self.capped_64x256_forecast = self.scaled_64x256_forecast.clip(-20, 20)
        divisor = TrendData.rule_count
        self.raw_combined_forecast = (
            self.capped_2x8_forecast * 1 / divisor
            + self.capped_4x16_forecast * 1 / divisor
            + self.capped_8x32_forecast * 1 / divisor
            + self.capped_16x64_forecast * 1 / divisor
            + self.capped_32x128_forecast * 1 / divisor
            + self.capped_64x256_forecast * 1 / divisor
        )
        self.scaled_combined_forecast = self.raw_combined_forecast * TrendData.fdm
        self.capped_combined_forecast = self.scaled_combined_forecast.clip(-20, 20)
        self.variableDict = TrendData.getVariableDict(self)

        print("Trend Class Initialized")

    # creates connection to postgres and specified database
    # drops table if exists and replaces for all attributes in trend
    def updateTrend(self, database):
        DB_PARAMS["dbname"] = database

        connection_str = f'postgresql://{DB_PARAMS["user"]}:{DB_PARAMS["password"]}@{DB_PARAMS["host"]}:{DB_PARAMS["port"]}/{DB_PARAMS["dbname"]}'

        engine = create_engine(connection_str)
        with engine.connect() as conn:
            for name, df in self.variableDict.items():
                if type(df) == pd.DataFrame:
                    basic_historical_append(df, name, conn)

            conn.close()

    # calculates the ewm for each individual contract after removing null days
    def calculate_ewm(self, df, spanParam):
        dfNew = pd.DataFrame()
        for column in df:
            series = df[column]
            filteredSeries = series.dropna()
            filteredSeries = filteredSeries.ewm(span=spanParam, min_periods=2).mean()
            filteredSeries.name = column
            dfNew = pd.concat([dfNew, filteredSeries], axis=1)

        dfNew = dfNew.sort_index(axis=0)

        return dfNew

    # gets the variable dictionary for all the attributes in trend
    def getVariableDict(self):
        dict = vars(self)
        return dict

    # gets dataframe of all of the close prices for each contract
    def getClose(self, trendDict):
        df = pd.DataFrame()
        for key, value in trendDict.items():
            data_dt = value
            index = data_dt.Date
            index = date_format_correction(index)
            close = data_dt.Close
            close.name = key
            close = close.set_axis(index)
            df = pd.concat([df, close], axis=1)

        df = df.sort_index(axis=0)
        # print(df)
        return df

    # gets the std dev of price daily
    def getStdevTrend(self, trendDict):
        df = pd.DataFrame()
        for key, value in trendDict.items():
            data_dt = value
            index = data_dt.Date
            index = date_format_correction(index)
            close = data_dt.Close
            unadj_close = data_dt.Unadj_Close
            close = close.set_axis(index)
            unadj_close = unadj_close.set_axis(index)

            stdDev = standardDeviation(close, unadj_close)
            stdev_price = stdDev.daily_risk_price_terms()

            stdev_price.name = key

            df = pd.concat([df, stdev_price], axis=1)

        df = df.sort_index(axis=0)

        return df


# converts df to a sql datatable with index labeled 'Date' and replaces existing table using engine parameter provided
def basic_historical_append(df, tableName, engine):

    print(df)

    df.to_sql(tableName, con=engine, index_label="Date", if_exists="replace")

    print("Appended", tableName, "to postgres")


def main():
    database = "Norton"
    if database == "Norton":
        dbTrend = "trend"
        dbCarry = "carry"
        contractParam = contracts_Norton
    elif database == "Bento":
        contractParam = contracts_databento
        dbTrend = "trend_values_databento"
        dbCarry = "carry_values_databento"

    trendDict, carryDict = getBothDict(contractParam, database)
    carry = CarryData(carryDict, trendDict)
    trend = TrendData(trendDict)

    trend.updateTrend(dbTrend)
    carry.updateCarry(dbCarry)

    # risk = RiskMeasures(trendDict)
    # risk.construct()


if __name__ == "__main__":
    main()


def potential_funtions():
    pass
    # #functioning for appending, working on, it works currently but takes too long for historical data
    # def append(dict, dbname):
    #     connection_str = f'postgresql://{DB_PARAMS["user"]}:{DB_PARAMS["password"]}@{DB_PARAMS["host"]}:{DB_PARAMS["port"]}/{DB_PARAMS["dbname"]}'
    #     engine = create_engine(connection_str)
    #
    #
    #     table = "practice"
    #     with engine.connect() as conn:
    #         for contract in dict.keys():
    #
    #             for date, value in dict[contract].items():
    #
    #                 sql = (f'INSERT INTO "{table}" ("date") SELECT (\'{date}\') WHERE NOT EXISTS (SELECT "date" FROM "{table}" WHERE "date" = \'{date}\');')
    #
    #                 result = conn.execute(text(sql))
    #                 conn.commit()
    #
    #
    #                 sql = f'UPDATE {table} SET "{contract}" = {value} WHERE "date" = \'{date}\' AND "{contract}" IS NULL;'
    #                 result = conn.execute(text(sql))
    #                 conn.commit()
    #         print(contract+" done!")


def oldMainCalls():
    pass

    # def individual_cross_dict_trend(ewma, contractParam):
    #     cross2x8 = {}
    #     cross4x16 = {}
    #     cross8x32 = {}
    #     cross16x64 = {}
    #     cross32x128 = {}
    #     cross64x256 = {}
    #
    #     for contract in contractParam:
    #         cross2x8[contract] = ewma[contract]['cross2x8']
    #         cross4x16[contract] = ewma[contract]['cross4x16']
    #         cross8x32[contract] = ewma[contract]['cross8x32']
    #         cross16x64[contract] = ewma[contract]['cross16x64']
    #         cross32x128[contract] = ewma[contract]['cross32x128']
    #         cross64x256[contract] = ewma[contract]['cross64x256']
    #
    #     return cross2x8, cross4x16, cross8x32, cross16x64, cross32x128, cross64x256

    # def individual_ewma_dict_trend(ewma, contractParam):
    #     ewma2 = {}
    #     ewma4 = {}
    #     ewma8 = {}
    #     ewma16 = {}
    #     ewma32 = {}
    #     ewma64 = {}
    #     ewma128 = {}
    #     ewma256 = {}
    #     for contract in contractParam:
    #         ewma2[contract] = ewma[contract]['ewm2']
    #         ewma4[contract] = ewma[contract]['ewm4']
    #         ewma8[contract] = ewma[contract]['ewm8']
    #         ewma16[contract] = ewma[contract]['ewm16']
    #         ewma32[contract] = ewma[contract]['ewm32']
    #         ewma64[contract] = ewma[contract]['ewm64']
    #         ewma128[contract] = ewma[contract]['ewm128']
    #         ewma256[contract] = ewma[contract]['ewm256']
    #
    #     return ewma2, ewma4, ewma8, ewma16, ewma32, ewma64, ewma128, ewma256
    #
    # def individual_ewma_dict_carry(smooth, contractParam):
    #     ewma5 = {}
    #     ewma20 = {}
    #     ewma60 = {}
    #     ewma120 = {}
    #     for contract in contractParam:
    #         ewma5[contract] = smooth[contract]['ewm5']
    #         ewma20[contract] = smooth[contract]['ewm20']
    #         ewma60[contract] = smooth[contract]['ewm60']
    #         ewma120[contract] = smooth[contract]['ewm120']
    #
    #     return ewma5, ewma20, ewma60, ewma120
    # choice = input("Trend (1), Carry(2), or Both(3): ")
    # if (choice == "1"):
    #     DB_PARAMS["dbname"] = 'trenddata'
    #     all_ewma, all_stdev, all_rawforecast = all_contracts_trend(contracts)
    #
    #     # basic_historical_append(all_stdev, 'trend', 'annualized-std-dev-price')
    #
    #     # cross2x8, cross4x16, cross8x32, cross16x64, cross32x128, cross64x256 = individual_cross_dict_trend(all_rawforecast, contracts)
    #
    #     # basic_historical_append(cross2x8, 'trend', 'raw-2x8-forecast')
    #     # basic_historical_append(cross4x16, 'trend', 'raw-4x16-forecast')
    #     # basic_historical_append(cross8x32, 'trend', 'raw-8x32-forecast')
    #     # basic_historical_append(cross16x64, 'trend', 'raw-16x64-forecast')
    #     # basic_historical_append(cross32x128, 'trend', 'raw-32x128-forecast')
    #     # basic_historical_append(cross64x256, 'trend', 'raw-64x256-forecast')
    #
    #
    #
    # elif (choice == "2"):
    #     DB_PARAMS["dbname"] = 'carrydata'
    #     all_price, all_carry, all_rawCarry, all_expiry_difference, all_ann_carry, all_stdev_carry, all_riskadj_carry, all_smooth_carry, all_scaled_carry, all_capped_carry, all_raw_combined_carry, all_scaled_forecast_carry, all_capped_forecast_carry = all_contracts_carry(
    #         contracts)
    #
    #
    #
    # elif (choice == "3"):
    #     DB_PARAMS["dbname"] = 'trenddata'
    #     all_ewma, all_stdev, all_rawforecast = all_contracts_trend(contracts)
    #     DB_PARAMS["dbname"] = 'carrydata'
    #     all_price, all_carry, all_rawCarry, all_expiry_difference, all_ann_carry, all_stdev_carry, all_riskadj_carry, all_smooth_carry, all_scaled_carry, all_capped_carry, all_raw_combined_carry, all_scaled_forecast_carry, all_capped_forecast_carry = all_contracts_carry(
    #         contracts)


def oldCarryFunctions():
    pass
    # #Accessor function to return the series of a carry contract
    # #Returns multiple series: Price, Carry, Raw Carry, Expiry Difference
    # def getPriceCarryRawExpire(dt):
    #     data_dt = get_data_from_db(dt)
    #     index = data_dt.Date
    #     index = date_format_correction(index)
    #     price = data_dt.Price
    #     price = price.set_axis(index)
    #     carry = data_dt.Carry
    #     carry = carry.set_axis(index)
    #     rawCarry = price-carry
    #
    #     price_date = data_dt.Price_Contract
    #     price_date = price_date.set_axis(index)
    #     carry_date = data_dt.Carry_Contract
    #     carry_date = carry_date.set_axis(index)
    #
    #     price_year_frac = _total_year_frac_from_contract_series(price_date)
    #     carry_year_frac = _total_year_frac_from_contract_series(carry_date)
    #     expiry_difference = carry_year_frac - price_year_frac
    #
    #     return price, carry, rawCarry, expiry_difference
    #
    # #Returns a series containing the fraction of a year for a contract
    #
    #
    # #Return the annualized stdev for carry, utilizing standardDeviation class
    # def calculate_stdev_carry(dt):
    #     DB_PARAMS["dbname"] = 'trenddata'
    #     name = dt+"_data"
    #     data_df = get_data_from_db(name)
    #     index = data_df.Date
    #     index = date_format_correction(index)
    #     close = data_df.Close
    #     unadj_close = data_df.Unadj_Close
    #     close = close.set_axis(index)
    #     unadj_close = unadj_close.set_axis(index)
    #
    #
    #     stdDev = standardDeviation(close, unadj_close)
    #     stdev_annual = stdDev.annual_risk_price_terms()
    #
    #     DB_PARAMS["dbname"] = 'carrydata'
    #     return stdev_annual
    #
    # #Turns each dictionary of PD Series into a DF and calculates the mean for the raw_combined
    # #Returns this average forecast as a series
    # def raw_combined_forecast(dict):
    #
    #     df = pd.DataFrame(dict)
    #     average_forecast = df.mean(axis=1)
    #
    #     return average_forecast
    #
    # #Calculates all values for carry all the way up to Capped Combined Forecast
    # #Returns each data value as a dictionary, keys are contract initials, values are series of the data as far back as historical data goes
    # def all_contracts_carry(contractParam):
    #     ewma_list = ["ewm5", "ewm20", "ewm60", "ewm120"]
    #     price_dict = {}
    #     carry_dict = {}
    #     rawCarry_dict = {}
    #     expiry_difference = {}
    #     ann_carry = {}
    #     carry_stdev = {}
    #     risk_adj_carry = {}
    #     smooth_carry = {}
    #     scaled_carry = {}
    #     capped_carry = {}
    #     raw_combined = {}
    #     scaled_forecast = {}
    #     capped_forecast = {}
    #     rule_count = len(ewma_list)
    #     FDM_DICT = {1: 1.0, 2: 1.02, 3: 1.03, 4: 1.04}
    #     fdm = FDM_DICT[rule_count]
    #
    #
    #     for contract in contractParam:
    #         name = contract+"_Data_Carry"
    #         individual_scale = {}
    #         individual_capped = {}
    #         price_dict[contract], carry_dict[contract], rawCarry_dict[contract], expiry_difference[contract] = getPriceCarryRawExpire(name)
    #         ann_carry[contract] = rawCarry_dict[contract]/expiry_difference[contract]
    #         carry_stdev[contract] = calculate_stdev_carry(contract)
    #         risk_adj_carry[contract] = ann_carry[contract].divide(carry_stdev[contract])
    #         smooth_carry[contract] = ewma(risk_adj_carry[contract])
    #         for ewm in ewma_list:
    #             individual_scale[ewm] = smooth_carry[contract][ewm]*30
    #             individual_capped[ewm] = individual_scale[ewm].clip(-20, 20)
    #
    #         scaled_carry[contract] = individual_scale
    #         capped_carry[contract] = individual_capped
    #         raw_combined[contract] = raw_combined_forecast(capped_carry[contract])
    #
    #         scaled_forecast[contract] = raw_combined[contract] * fdm
    #         capped_forecast[contract] = scaled_forecast[contract].clip(-20, 20)
    #
    #     print("Carry Calculated")
    #
    #     return price_dict, carry_dict, rawCarry_dict, expiry_difference, ann_carry, carry_stdev, risk_adj_carry, smooth_carry, scaled_carry, capped_carry, raw_combined, scaled_forecast, capped_forecast


def oldTrendFunctions():
    pass

    # # Returns a dictionary of ewma's depending on the current database
    # def ewma(dt):
    #
    #     if DB_PARAMS["dbname"] == 'trenddata':
    #         data_df = get_data_from_db(dt)
    #         index = data_df.Date
    #         index = date_format_correction(index)
    #         close = data_df.Close
    #         close = close.set_axis(index)
    #         spans = [2, 4, 8, 16, 32, 64, 128, 256]
    #         ewma = {'ewm2': 0, 'ewm4': 0, 'ewm8': 0, 'ewm16': 0, 'ewm32': 0, 'ewm64': 0, 'ewm128': 0, 'ewm256': 0}
    #     elif DB_PARAMS["dbname"] == 'carrydata':
    #         close = dt
    #         spans = [5, 20, 60, 120]
    #         ewma = {'ewm5': 0, 'ewm20': 0, 'ewm60': 0, 'ewm120': 0}
    #
    #     i = 0
    #     for ewm_ in ewma.keys():
    #         if DB_PARAMS["dbname"] == 'trenddata':
    #             ewma[ewm_] = close.ewm(span=spans[i], min_periods=2).mean()
    #         elif DB_PARAMS["dbname"] == 'carrydata':
    #             ewma[ewm_] = close.ewm(span=spans[i]).mean()
    #         i += 1
    #
    #     return ewma
    #
    # # Returns a Series of the daily stdev of price utilizing standardDeviation class
    # def calculate_stdev_risk(dt):
    #     data_df = get_data_from_db(dt)
    #     index = data_df.Date
    #     index = date_format_correction(index)
    #     close = data_df.Close
    #     unadj_close = data_df.Unadj_Close
    #     close = close.set_axis(index)
    #     unadj_close = unadj_close.set_axis(index)
    #
    #     stdDev = standardDeviation(close, unadj_close)
    #     stdev_price = stdDev.daily_risk_price_terms()
    #
    #     return stdev_price
    #
    # # Returns the rawforecast between the fast and slow spans, divided by the stdev
    # def calculate_rawforecast(fast_span, slow_span, stdev):
    #     crossover = fast_span - slow_span
    #     raw_forecast = crossover / stdev
    #
    #     return raw_forecast
    #
    # # Calculates all values up until raw forecast for trend
    # def all_contracts_trend(contractsPara):
    #     ewma_list = ['ewm2', 'ewm4', 'ewm8', 'ewm16', 'ewm32', 'ewm64', 'ewm128', 'ewm256']
    #     all_ewma_dict = {}
    #     all_stdev_dict = {}
    #     all_rawforecast = {}
    #     for contract in contractsPara:
    #         name = contract + "_data"
    #         individual_rawforecast = {'cross2x8': 0, 'cross4x16': 0, 'cross8x32': 0, 'cross16x64': 0, 'cross32x128': 0,
    #                                   'cross64x256': 0}
    #         all_ewma_dict[contract] = ewma(name)
    #         all_stdev_dict[contract] = calculate_stdev_risk(name)
    #         x = 0
    #         for cross in individual_rawforecast.keys():
    #             rawforecast = calculate_rawforecast(all_ewma_dict[contract][ewma_list[x]],
    #                                                 all_ewma_dict[contract][ewma_list[x + 2]], all_stdev_dict[contract])
    #             individual_rawforecast[cross] = rawforecast
    #             x += 1
    #         all_rawforecast[contract] = individual_rawforecast
    #
    #     print("Trend Calculated")
    #
    #     return all_ewma_dict, all_stdev_dict, all_rawforecast
