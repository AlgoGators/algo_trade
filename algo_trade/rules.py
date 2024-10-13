from functools import cache
from typing import Callable, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd # type: ignore

from algo_trade.instrument import Future, Instrument
from algo_trade.risk_measures import RiskMeasure, StandardDeviation
from algo_trade._constants import DAYS_IN_YEAR

Rule = Callable[[], pd.DataFrame]

T = TypeVar('T', bound=Instrument)

def capital_scaling(
        instruments: list[Future],
        capital: float
    ) -> pd.DataFrame:

    df : pd.DataFrame = pd.DataFrame()
    for instrument in instruments:
        if df.empty:
            df = (
                instrument.front
                .get_close()
                .to_frame(instrument.name)
            )
        else:
            df = df.join(
                instrument.front
                .get_close()
                .to_frame(instrument.name),
                how="outer"
            )

    df.ffill(inplace=True)

    capital_weighting : pd.DataFrame = capital / df

    return capital_weighting

def risk_parity(
        risk_object: RiskMeasure[T]
    ) -> pd.DataFrame:

    if risk_object.tau is None:
        raise ValueError("Risk Measure object must have a tau value")

    std : StandardDeviation = risk_object.get_var().to_standard_deviation().annualize()

    return risk_object.tau / std

def equal_weight(
        instruments: list[Future]
    ) -> pd.DataFrame:

    df : pd.DataFrame = pd.DataFrame()
    for instrument in instruments:
        if df.empty:
            df = (
                instrument.front
                .get_close()
                .to_frame(instrument.name)
            )
        else:
            df = df.join(
                instrument.front
                .get_close()
                .to_frame(instrument.name),
                how="outer"
            )

    df.ffill(inplace=True)

    not_null_mask : pd.DataFrame = df.notnull()
    weight_mask : pd.DataFrame = 1 / df.notnull().sum(axis=1).astype(int)

    weights : pd.DataFrame = not_null_mask.mul(weight_mask, axis=0)

    return weights

@cache
def exponential_weight_matrix(n : int, N : int) -> np.ndarray:
    weight_matrix_n : np.ndarray = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i >= j:
                weight_matrix_n[i, j] = (2 / (n + 1)) ** (i - j) * (2 / (n + 1))

    return weight_matrix_n

def raw_forecast(
    speed : int,
    prices : pd.DataFrame,
    annualized_std : pd.DataFrame
    ) -> npt.NDArray[np.float64]:

    prices_matrix : np.ndarray = prices.to_numpy(dtype=np.float64)
    std_matrix : np.ndarray = annualized_std.to_numpy(dtype=np.float64)

    std_price : np.ndarray = np.array(prices_matrix * std_matrix)

    N : int = prices_matrix.shape[0]

    weight_matrix_n : np.ndarray = exponential_weight_matrix(speed, N)
    weight_matrix_4n : np.ndarray = exponential_weight_matrix(4 * speed, N)

    y_n : np.ndarray = weight_matrix_n @ prices_matrix
    y_4n : np.ndarray = weight_matrix_4n @ prices_matrix

    trend : np.ndarray = np.array(y_n - y_4n)

    normalized_trend : np.ndarray = trend / std_price

    return normalized_trend

def regime_scaling(
    n : int,
    risk_object : RiskMeasure[Future]
    ) -> npt.NDArray[np.float64]:

    rolling_vol_means : pd.DataFrame = (
        risk_object
        .get_var()
        .to_standard_deviation()
        .to_frame()
        .rolling(window=DAYS_IN_YEAR*10, min_periods=DAYS_IN_YEAR)
        .mean()
    )

    relative_vol : pd.DataFrame = (
        risk_object
        .get_var()
        .to_standard_deviation()
        .to_frame()
        .div(rolling_vol_means)
    )

    quantiles : pd.DataFrame = relative_vol.apply(lambda x : x.rank() / x.count(), axis=0)

    quantiles = quantiles.where(pd.notnull(quantiles), 0)

    quantiles = quantiles.astype(np.float64)

    N : int = len(quantiles)

    weight_matrix_n : np.ndarray = exponential_weight_matrix(n, N)

    quantiles_np : np.ndarray = quantiles.to_numpy(np.float64)

    smoothed_quantiles : np.ndarray = weight_matrix_n @ quantiles_np

    multipliers : np.ndarray = 2 * np.ones((N, 1)) - 1.5 * smoothed_quantiles

    return multipliers

def fdm(
    weights : np.ndarray,
    forecasts : list[pd.DataFrame]
    ) -> npt.NDArray[np.float64]:

    if not forecasts:
        raise ValueError("At least one forecast is required")

    flattened_forecasts = [forecast.to_numpy(dtype=np.float64).flatten() for forecast in forecasts]

    forecast_df = pd.DataFrame(np.column_stack(flattened_forecasts))

    # calculate FDM every year
    correlation_matrices : list[pd.DataFrame] = []

    for i in range(0, len(forecast_df), DAYS_IN_YEAR):
        correlation_matrix = forecast_df.iloc[:i+DAYS_IN_YEAR].corr().clip(lower=0)
        correlation_matrices.append(correlation_matrix)

    FDMs_lst : list[np.ndarray] = []
    for correlation_matrix in correlation_matrices:
        FDM = 1 / np.sqrt(weights.T @ correlation_matrix.to_numpy(dtype=np.float64) @ weights).item()
        FDMs_lst.append(np.ones((256, 1), dtype=np.float64) * FDM)

    FDMs : np.ndarray = np.concatenate([FDMs_lst[0], *FDMs_lst])

    return FDMs[:len(forecast_df)]

def regime_scaled_trend(speed : int, risk_object : RiskMeasure, futures : list[Future]) -> pd.DataFrame:
    annualized_std = risk_object.get_var().to_standard_deviation().annualize().to_frame()
    
    prices : pd.DataFrame = pd.DataFrame()

    for future in futures:
        if prices.empty:
            prices = future.front.backadjusted.to_frame(future.name)
        else:
            prices = prices.join(future.front.backadjusted.to_frame(future.name), how="outer")

    prices.dropna(inplace=True)
    annualized_std.dropna(inplace=True)

    intersection = prices.index.intersection(annualized_std.index)
    prices = prices.loc[intersection]
    annualized_std = annualized_std.loc[intersection]

    raw_F : np.ndarray = raw_forecast(speed, prices, annualized_std)
    regime_scalar : np.ndarray = regime_scaling(10, risk_object)

    forecast : pd.DataFrame = pd.DataFrame(raw_F * regime_scalar, columns=[future.name for future in futures], index=intersection)
    normalized_forecast : pd.DataFrame = forecast.div(forecast.abs().mean().mean())

    return normalized_forecast

def trend_signals(
        trend_function : Callable[[int, RiskMeasure, list[Future]], pd.DataFrame],
        risk_object : RiskMeasure[Future],
        futures : list[Future],
        speeds : list[int],
        bounds : tuple[int, int] = (-2, 2)
    ) -> pd.DataFrame:

    forecasts : list[pd.DataFrame] = [trend_function(speed, risk_object, futures).clip(bounds[0], bounds[1]) for speed in speeds]

    weights : np.ndarray = np.ones((len(speeds), 1), dtype=np.float64) / len(forecasts)

    FDM : np.ndarray = fdm(weights, forecasts)

    average_forecasts = pd.DataFrame(0, index=forecasts[0].index, columns=[future.name for future in futures])

    for forecast in forecasts:
        average_forecasts += forecast

    average_forecasts /= len(forecasts)

    signals = (FDM * average_forecasts).clip(bounds[0], bounds[1])

    return signals

def IDM(risk_object : RiskMeasure[Future]) -> pd.DataFrame:
    """ IDM = 1 / √(w.ρ.wᵀ) where w is the weight vector and ρ is the correlation matrix """

    returns = risk_object.get_returns()

    weights = equal_weight(risk_object.instruments)

    rolling_corrs = returns.rolling(window=DAYS_IN_YEAR, min_periods=DAYS_IN_YEAR).corr()

    IDMs = pd.Series(index=returns.index)

    # Group by the date to get each correlation matrix
    for date, corr_matrix in rolling_corrs.groupby(level=0):
        # Drop the date index level to work with the matrix
        corr_matrix = corr_matrix.droplevel(0)

        IDMs.loc[date] = 1 / np.sqrt(weights.loc[date].values.T @ corr_matrix @ weights.loc[date].values)

    IDMs = IDMs.bfill()

    return pd.concat([IDMs.rename(instrument_name) for instrument_name in returns.columns], axis=1)
