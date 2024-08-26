import pandas as pd
import numpy as np
from abc import ABC

from algo_trade.instrument import Instrument, Future

class _utils:
    def ffill_zero(df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward fill zeros in a DataFrame. This function will replace all zeros with the last non-zero value in the DataFrame.
        """
        # We assume any gaps in percent returns at this point are because the market was closed that day,
        # but only fill forward;
        # Find the index of the first non-NaN value in each column
        first_non_nan_index = df.apply(lambda x: x.first_valid_index())

        # Iterate over each column and replace NaN values below the first non-NaN index with 0
        for column in df.columns:
            first_index = first_non_nan_index[column]
            df.loc[first_index:, column] = df.loc[first_index:, column].fillna(0)

        return df

class RiskMeasure(ABC):
    def __init__(self) -> None:
        pass

    def get_returns(self) -> pd.DataFrame:
        raise NotImplementedError("Method not implemented")

    def get_product_returns(self) -> pd.DataFrame:
        raise NotImplementedError("Method not implemented")

    def get_var(self) -> pd.DataFrame:
        raise NotImplementedError("Method not implemented")

    def get_cov(self) -> pd.DataFrame:
        raise NotImplementedError("Method not implemented")

class GARCH(RiskMeasure):
    def __init__(
        self,
        instruments : list[Instrument],
        weights : tuple[float, float, float],
        minimum_observations : int,
        fill : bool = True) -> None:

        self.instruments = instruments
        self.weights = weights
        self.minimum_observations = minimum_observations
        self.fill = fill

        self.__returns = pd.DataFrame()
        self.__product_returns = pd.DataFrame()
        self.__var = pd.DataFrame()
        self.__cov = pd.DataFrame()

    def get_returns(self) -> pd.DataFrame:
        if not self.__returns.empty:
            return self.__returns

        if not all(isinstance(instrument, Future) for instrument in self.instruments):
            raise NotImplementedError("Only futures are supported")

        instrument : Future
        for instrument in self.instruments:
            backadjusted_prices : pd.Series = instrument.price
            unadjusted_prices : pd.Series = instrument.front.get_close()

            #* For equation see: 
            # https://qoppac.blogspot.com/2023/02/percentage-or-price-differences-when.html
            percent_change : pd.Series = (
                backadjusted_prices - backadjusted_prices.shift(1)) / unadjusted_prices.shift(1)

            percent_change.name = instrument.name
            self.__returns = pd.concat([self.__returns, percent_change], axis=1)

        if self.fill:
            self.__returns = _utils.ffill_zero(self.__returns)

        return self.__returns
    
    def get_product_returns(self) -> pd.DataFrame:
        if not self.__product_returns.empty:
            return self.__product_returns

        returns = self.get_returns()

        product_dictionary : dict[str, pd.Series] = {}

        for i, instrument_i in enumerate(self.instruments):
            for j, instrument_j in enumerate(self.instruments):
                if i > j:
                    continue
                
                product_dictionary[f'{instrument_i.name}_{instrument_j.name}'] = returns[instrument_i.name] * returns[instrument_j.name]

        self.__product_returns = pd.DataFrame(product_dictionary, index=returns.index)

        self.__product_returns = _utils.ffill_zero(self.__product_returns) if self.fill else self.__product_returns

        return self.__product_returns

    def get_var(self) -> pd.DataFrame:
        if not self.__var.empty:
            return self.__var[self.minimum_observations:]
        
        self.__var : pd.DataFrame = pd.DataFrame()

        for i, instrument in enumerate(self.__returns.columns.tolist()):
            squared_returns = self.__returns[instrument] ** 2
            squared_returns.dropna(inplace=True)

            dates = squared_returns.index

            # Calculate rolling LT variance
            LT_variances = squared_returns.rolling(window=self.minimum_observations).mean().bfill()

            df = pd.Series(index=dates)
            df.iloc[0] = squared_returns.iloc[0]

            for j, _ in enumerate(dates[1:], 1):
                df.iloc[j] = squared_returns.iloc[j] * self.weights[0] + df.iloc[j-1] * self.weights[1] + LT_variances.iloc[j] * self.weights[2]

            if i == 0:
                self.__var = df.to_frame(instrument)
                continue

            self.__var = pd.merge(self.__var, df.to_frame(instrument), how='outer', left_index=True, right_index=True)

        self.__var = self.__var.interpolate() if self.fill else self.__var

        return self.__var[self.minimum_observations:]

    def get_cov(self) -> pd.DataFrame:
        if not self.__cov.empty:
            return self.__cov.iloc[self.minimum_observations:, :]

        product_returns : np.ndarray = self.get_product_returns().dropna().values
        LT_covariances : np.ndarray = self.get_product_returns().rolling(window=self.minimum_observations).mean().bfill().values

        self.__cov = pd.DataFrame(index=self.get_product_returns().index, columns=self.get_product_returns().columns, dtype=float)
        self.__cov.iloc[0] = product_returns[0]

        for i in range(1, len(product_returns)):
            self.__cov.iloc[i] = product_returns[i] * self.weights[0] + self.__cov.iloc[i-1] * self.weights[1] + LT_covariances[i] * self.weights[2]

        self.__cov = self.__cov.interpolate() if self.fill else self.__cov

        return self.__cov.iloc[self.minimum_observations:, :]
