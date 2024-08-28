import pandas as pd
import numpy as np
from abc import ABC
from typing import Self, Optional

from algo_trade.instrument import Instrument, Future
from algo_trade._constants import DAYS_IN_YEAR

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

class StandardDeviation(pd.DataFrame):
    def __init__(self, data : pd.DataFrame = None) -> None:
        super().__init__(data)
        self.__is_annualized : bool = False

    def annualize(self, inplace=False) -> Optional[Self]:
        if self.__is_annualized:
            return self

        factor : float = DAYS_IN_YEAR ** 0.5

        if inplace:
            self *= factor
            self.__is_annualized = True
            return None

        new = StandardDeviation(self)
        new.annualize(inplace=True)
        return new

    def to_variance(self) -> 'Variance':
        return Variance(self ** 2)
    
    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self)

class Variance(pd.DataFrame):
    def __init__(self, data : pd.DataFrame = None) -> None:
        super().__init__(data)
        self.__is_annualized = False

    def annualize(self, inplace=False) -> Optional[Self]:
        if self.__is_annualized:
            return self

        factor : float = DAYS_IN_YEAR

        if inplace:
            self *= factor
            self.__is_annualized = True
            return None

        new = Variance(self)
        new = new.annualize(inplace=True)
        return new
    
    def to_standard_deviation(self) -> 'StandardDeviation':
        return StandardDeviation(self ** 0.5)
    
    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self)

class RiskMeasure(ABC):
    def __init__(self, tau : float = None) -> None:
        if tau is not None:
            self.tau = tau

    @property
    def tau(self) -> float:
        if not hasattr(self, '_tau'):
            raise ValueError("tau is not set")
        return self._tau
    
    @tau.setter
    def tau(self, value : float) -> None:
        if (value < 0) or not isinstance(value, float):
            raise ValueError("tau, x, is a float such that x ∈ (0, inf)")
        self._tau = value

    def get_returns(self) -> pd.DataFrame:
        raise NotImplementedError("Method not implemented")

    def get_product_returns(self) -> pd.DataFrame:
        raise NotImplementedError("Method not implemented")

    def get_var(self) -> Variance:
        raise NotImplementedError("Method not implemented")

    def get_cov(self) -> pd.DataFrame:
        raise NotImplementedError("Method not implemented")
    
    def get_jump_cov(self, percentile : float, window : int) -> pd.DataFrame:
        if (percentile < 0) or (percentile > 1):
            raise ValueError("percentile, x, is a float such that x ∈ (0, 1)")

        dates = self.get_cov().index

        jump_covariances = pd.DataFrame(index=dates, columns=self.get_cov().columns)

        for i in range(len(dates)):
            if i < window:
                continue

            window_covariances = self.get_cov().iloc[i-window:i]
            jump_covariances.iloc[i] = window_covariances.quantile(percentile)

        return jump_covariances

class GARCH(RiskMeasure):
    def __init__(
        self,
        risk_target : float,
        instruments : list[Instrument],
        weights : tuple[float, float, float],
        minimum_observations : int,
        fill : bool = True) -> None:

        super().__init__(tau=risk_target)

        self.instruments = instruments
        self.weights = weights
        self.minimum_observations = minimum_observations
        self.fill = fill

        self.__returns = pd.DataFrame()
        self.__product_returns = pd.DataFrame()
        self.__var = Variance()
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

    def get_var(self) -> Variance:
        if not self.__var.empty:
            return self.__var
        
        variance : pd.DataFrame = pd.DataFrame()

        for i, instrument in enumerate(self.get_returns().columns.tolist()):
            squared_returns = self.get_returns()[instrument] ** 2
            squared_returns.dropna(inplace=True)

            dates = squared_returns.index

            # Calculate rolling LT variance
            LT_variances = squared_returns.rolling(window=self.minimum_observations).mean().bfill()

            df = pd.Series(index=dates)
            df.iloc[0] = squared_returns.iloc[0]

            for j, _ in enumerate(dates[1:], 1):
                df.iloc[j] = squared_returns.iloc[j] * self.weights[0] + df.iloc[j-1] * self.weights[1] + LT_variances.iloc[j] * self.weights[2]

            if i == 0:
                variance = df.to_frame(instrument)
                continue

            variance = pd.merge(variance, df.to_frame(instrument), how='outer', left_index=True, right_index=True)

        variance = variance.interpolate() if self.fill else variance

        self.__var : Variance = Variance(variance[self.minimum_observations:])

        return self.__var

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

    def get_jump_cov(self, percentile : float, window : int) -> pd.DataFrame:
        return super().get_jump_cov(percentile=percentile, window=window)