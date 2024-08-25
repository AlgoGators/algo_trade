import pandas as pd
import numpy as np

from instrument import Instrument, Future

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

class RiskMeasures:
    def __init__(
        self,
        instruments : list[Instrument],
        weights : list[tuple[float, float, float]],
        minimum_observations : int,
        fill : bool = True) -> None:

        self.instruments = instruments
        self.weights = weights
        self.minimum_observations = minimum_observations
        self.fill = fill

    def calculate_returns(self) -> pd.DataFrame:
        daily_returns : pd.DataFrame = pd.DataFrame()
        instrument : Instrument

        if type(Instrument) != Future:
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
            daily_returns = pd.concat([daily_returns, percent_change], axis=1)

        if self.fill:
            daily_returns = _utils.ffill_zero(daily_returns)

        return daily_returns

