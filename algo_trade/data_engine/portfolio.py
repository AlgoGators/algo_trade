import asyncio
import os
from datetime import datetime, timedelta
from typing import Any, Dict

import databento as db
import pandas as pd
import toml
from tqdm import tqdm

from .future import Historical, Live

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
config_dir = os.path.join(base_dir, 'config')
config_path = os.path.join(config_dir, 'config.toml')

config: Dict[str, Any] = toml.load(config_path)

class Portfolio:
    """
    Portfolio:
    The Portfolio class is responsible for managing the contracts and each individual contract object
    It is the macro object used in the data pipeline to manage the contracts and their data
    """

    def __init__(self, instrument_list: list):
        # Setting universal start and end date for the portfolio
        # Reading the contracts.csv to find the datasets and symbols
        df = pd.read_csv("contracts.csv", index_col="Data Symbol")
        df.dropna(inplace=True)
        self.symbols = []
        avail_list = df.index.tolist()
        for symbol in instrument_list:
            if symbol not in avail_list:
                raise ValueError(f"Symbol {symbol} not found in contracts.csv")
            else:
                self.symbols.append(symbol)

        self.datasets = {symbol: df.loc[symbol]["Data Set"] for symbol in self.symbols}


class HistoricalPortfolio(Portfolio):
    def __init__(self, instrument_list: list, start_date, end_date, full=False):
        super().__init__(instrument_list)

        if full:
            # Find range of the dataset and construct the contracts
            self.contracts, client = self.assemble_full()
            # Begin batch downloading the data
            self.submit_jobs(client)
            self.download_data(client)
        else:
            self.start_date = start_date
            self.end_date = end_date
            # Assemble all the contracts into a dictionary but do not make any async calls
            self.contracts = self.assemble()
            asyncio.run(self.build())
            # Build the contracts by aggregating the data and definitions coroutines and calling each
            # Update existing parquet files with the latest data

    def assemble(self) -> dict:
        # Create a dictionary of each contract and its respective Contract object
        # Wait=True is used to ensure no async calls are made
        contracts = {}
        for symbol, dataset in self.datasets.items():
            contracts[symbol] = Historical(
                symbol, dataset, self.start_date, self.end_date, wait=True
            )
        return contracts

    def assemble_full(self) -> dict:
        # Create a dictionary of each contract and its respective Contract object
        # Wait=True is used to ensure no async calls are made
        client = db.Historical(config["databento"]["api_historical"])
        # Find each unique dataset in the contracts file
        unique_datasets = list(set(self.datasets.values()))
        # Build a dictionary with a key of the dataset and a value of the range of the dataset(another dictionary with keys of start_date and end_date)
        ranges = {
            dataset: client.metadata.get_dataset_range(dataset)
            for dataset in unique_datasets
        }
        # Assemble the contracts
        contracts = {}
        for symbol, dataset in self.datasets.items():
            contracts[symbol] = Historical(
                symbol,
                dataset,
                ranges[dataset]["start"],
                ranges[dataset]["end"],
                wait=True,
            )

        # Return the contracts
        return contracts, client

    def get_contract(self, symbol: str) -> Historical:
        # Returns the contract object for the given symbol
        return self.contracts[symbol]

    def get_all_contracts(self) -> dict:
        # Returns all the contracts in the portfolio
        return self.contracts

    async def build(self):
        # Progress bar
        pbar = tqdm(
            total=len(self.contracts), desc="Building Contracts", unit="contracts"
        )

        # Collect all the coroutines and run them
        async def build_and_update(contract):
            result = await contract.build()
            pbar.update()
            return result

        # Run the coroutines
        coroutines = [
            build_and_update(contract) for contract in self.contracts.values()
        ]
        results = await asyncio.gather(*coroutines)

        # Close the progress bar
        pbar.close()

        # Return the results
        return results

    def submit_jobs(self, client):
        pbar = tqdm(total=len(self.contracts), desc="Submitting Jobs", unit="contracts")
        # Begin subbmitting batch download requests per contract
        self.jobs = {}
        # ISO 8601 format for the current time
        self.beggining = datetime.now().isoformat()
        for symbol, contract in self.contracts.items():
            try:
                # Begin submitting jobs to the batch download
                contract_with_rolls = [
                    f"{contract.symbol}.c.0",
                    f"{contract.symbol}.c.1",
                ]
                data_job = client.batch.submit_job(
                    dataset=contract.dataset,
                    symbols=contract_with_rolls,
                    schema="OHLCV-1d",
                    encoding="dbn",
                    compression="zstd",
                    start=contract.start,
                    end=contract.end,
                    stype_in="continuous",
                    split_duration="none",
                )
                def_job = client.batch.submit_job(
                    dataset=contract.dataset,
                    symbols=contract_with_rolls,
                    schema="definition",
                    encoding="dbn",
                    compression="zstd",
                    start=contract.start,
                    end=contract.end,
                    stype_in="continuous",
                    split_duration="none",
                )
                self.jobs[contract.symbol] = (data_job, def_job)
                # Wait for 5 seconds before submitting the next job to avoid rate limiting
                pbar.update()
                asyncio.run(asyncio.sleep(6))
            except db.BentoClientError as e:
                print(f"Error submitting job for {contract.symbol}: {e}")
        pbar.close()
        # Now we have submitted all the jobs, we can begin polling the jobs

    def download_data(self, client):
        # Begin polling the jobs and checking last jobs status
        time_elapsed = 0
        wait_time = 120
        asyncio.run(asyncio.sleep(20))
        while True:
            # Check if all jobs are complete, returns a list of dictionaries
            if (
                client.batch.list_jobs(
                    since=self.beggining, states=["received", "queued", "processing"]
                )
                == []
            ):
                break
            else:
                # Sleep for 60 seconds
                print(f"Waiting for jobs to complete... {time_elapsed} seconds elapsed")
                time_elapsed += wait_time
                asyncio.run(asyncio.sleep(wait_time))
        pbar = tqdm(total=len(self.jobs), desc="Downloading Data", unit="contracts")
        for symbol, jobs in self.jobs.items():
            # Wait for 3 seconds before downloading the data to avoid rate limiting
            asyncio.run(asyncio.sleep(3))
            data_job, def_job = jobs
            data_paths = client.batch.download(
                output_dir="tmp",
                job_id=data_job["id"],
            )
            def_paths = client.batch.download(
                output_dir="tmp",
                job_id=def_job["id"],
            )
            # Load the data and definitions into a DBNStore object
            data_dbn = db.DBNStore.from_file([p for p in data_paths if p.suffix == '.zst'][0])
            def_dbn = db.DBNStore.from_file([p for p in def_paths if p.suffix == '.zst'][0])
            # Merge the data and definitions dataframes and store them
            data = data_dbn.to_df()
            definitions = def_dbn.to_df()
            # raw = pd.merge(
            #     data,
            #     definitions,
            #     how="inner",
            #     left_index=True,
            #     right_on=definitions.index,
            #     suffixes=("_data", "_definition"),
            # )
            data.index.name = "timestamp"
            definitions.index.name = "timestamp"
            data.to_parquet(f"raw/{symbol}_data.parquet")
            definitions.to_parquet(f"raw/{symbol}_definitions.parquet")
            raw = pd.merge_asof(
                data,
                definitions,
                on="timestamp",
                direction="nearest",
                suffixes=("_data", "_definition"),
            )
            raw.to_parquet(f"raw/{symbol}_full.parquet")
            # Update progress bar
            pbar.update()


class LivePortfolio(Portfolio):
    def __init__(self, instrument_list: list):
        super().__init__(instrument_list)
        """ 
        LivePortfolio:
        The LivePortfolio class is responsible for updating existing parquet files with the latest data based on the current date and the last date in the parquet file
        We open each {symbol}_full.parquet file and check the last date in the file, then we pull data from databento with the start date as the last date in the parquet file and the end date as the current date
        We then append the new data to the parquet file
        """
        # Assemble all the contracts into a dictionary but do not make any async calls
        self.contracts = self.assemble()
        # Build the contracts by aggregating the data and definitions coroutines and calling each
        asyncio.run(self.build())
        # Update existing parquet files with the latest data
        self.update_data()

    def assemble(self) -> dict:
        # Create a dictionary of each contract and its respective Contract object
        # This function does most of the heavy lifting in the Portfolio class
        contracts = {}
        for symbol, dataset in self.datasets.items():
            """
            We open each {symbol}_full.parquet file in the raw directory and check the last date in the file
            """
            try:
                df = pd.read_parquet(f"raw/{symbol}_full.parquet")
                last_date = df['timestamp'].iloc[-1]
            except FileNotFoundError:
                # Set the last date yesterday if the file is not found
                last_date = datetime.now() - timedelta(days=1)
                print(f"File not found for {symbol}")
            contracts[symbol] = Historical(
                symbol,
                dataset,
                last_date,
                datetime.now() - timedelta(days=1),
                wait=True,
            )
        return contracts

    async def build(self):
        # Progress bar
        pbar = tqdm(
            total=len(self.contracts), desc="Building Contracts", unit="contracts"
        )

        # Collect all the coroutines and run them
        async def build_and_update(contract):
            result = await contract.build()
            pbar.update()
            return result

        # Run the coroutines
        coroutines = [
            build_and_update(contract) for contract in self.contracts.values()
        ]
        results = await asyncio.gather(*coroutines)

        # Close the progress bar
        pbar.close()

        # Return the results
        return results

    def update_data(self):
        """
        Update Data:
        Using the existing parquet files, we update the data with the latest data from databento.
        1. Load the existing parquet file
        2. Load the latest data from our contract dictionary
        3. Append and join the dataframes but do not duplicate the data
        """
        pbar = tqdm(total=len(self.contracts), desc="Updating Data", unit="contracts")
        for symbol, contract in self.contracts.items():
            try:
                # Load the existing parquet file
                df = pd.read_parquet(f"raw/{symbol}_full.parquet")
                data = pd.read_parquet(f"raw/{symbol}_data.parquet")
                definitions = pd.read_parquet(f"raw/{symbol}_definitions.parquet")
                # Load the latest data from our contract dictionary as well as our data and definitions
                new_data = contract.raw
                contract_data = contract.data.to_df()
                contract_def = contract.definitions.to_df() 
                # Append and join the dataframes but do not duplicate the data
                df = pd.concat([df, new_data], axis=0)
                data = pd.concat([data, contract_data], axis=0)
                definitions = pd.concat([definitions, contract_def], axis=0)
                # Drop duplicates based on the index
                df = df[~df.index.duplicated(keep="first")]
                data = data[~data.index.duplicated(keep="first")]
                definitions = definitions[~definitions.index.duplicated(keep="first")]
                # Save the updated dataframe
                df.to_parquet(f"raw/{symbol}_full.parquet")
                data.to_parquet(f"raw/{symbol}_data.parquet")
                definitions.to_parquet(f"raw/{symbol}_definitions.parquet")
            except FileNotFoundError:
                print(f"File not found for {symbol}")
            pbar.update()
