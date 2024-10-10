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

    std : StandardDeviation = risk_object.get_var().to_standard_deviation()
    std.annualize(inplace=True)

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

def exponential_weight_matrix(n : int, N : int) -> np.ndarray:
    weight_matrix_n : np.ndarray = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i >= j:
                weight_matrix_n[i, j] = (2 / (n + 1)) ** (i - j) * (2 / (n + 1))

    return weight_matrix_n

def forecast(
    speed : int,
    futures : list[Future],
    risk_object : RiskMeasure[Future]) -> npt.NDArray[np.float64]:

    std : pd.DataFrame = risk_object.get_var().to_standard_deviation().annualize().to_frame()
    prices : pd.DataFrame = pd.DataFrame()

    for future in futures:
        if prices.empty:
            prices = future.front.backadjusted
        else:
            prices = prices.join(future.front.backadjusted)

    prices.dropna(inplace=True)
    std.dropna(inplace=True)

    prices_matrix : np.ndarray = prices.to_numpy()
    std_matrix : np.ndarray = std.to_numpy()

    std_price = np.array(prices_matrix * std_matrix)

    N = len(prices)

    weight_matrix_n = exponential_weight_matrix(speed, N)
    weight_matrix_4n = exponential_weight_matrix(4 * speed, N)

    y_n = prices_matrix @ weight_matrix_n
    y_4n = prices_matrix @ weight_matrix_4n

    trend = np.array(y_n - y_4n)

    normalized_trend = trend / std_price

    return normalized_trend

def raw_forecast(
    n : int,
    futures : list[Future],
    risk_object : RiskMeasure[Future]
    ) -> npt.NDArray[np.float64]:

    std : StandardDeviation = risk_object.get_var().to_standard_deviation().annualize()

    normalized_trend = pd.DataFrame(dtype=np.float64, index=std.index) #np.zeros((len(futures), len(futures[0].front.backadjusted)))

    for future in futures:
        std_price_intersection = std[future.name].index.intersection(future.front.backadjusted.index)
        prices = np.array(future.front.backadjusted.loc[std_price_intersection])
        std_array = np.array(std[future.name].loc[std_price_intersection])

        N = len(prices)

        weight_matrix_n : np.ndarray = exponential_weight_matrix(n, N) #np.zeros((N, N))
        weight_matrix_4n : np.ndarray = exponential_weight_matrix(4*n, N) #np.zeros((N, N))

        y_n = prices @ weight_matrix_n
        y_4n = prices @ weight_matrix_4n

        trend : np.ndarray = np.array(y_n - y_4n)

        normalized_for_vol = trend / np.array(std_array * prices)

        normalized_trend.loc[std_price_intersection, future.name] = normalized_for_vol

    normalized_trend_bfill_np : np.ndarray = normalized_trend.bfill().to_numpy(np.float64)

    return normalized_trend_bfill_np

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

    quantiles_np = quantiles.to_numpy(np.float64)

    smoothed_quantiles = weight_matrix_n @ quantiles_np

    multipliers : np.ndarray = 2 * np.ones((N, 1)) - 1.5 * smoothed_quantiles

    return multipliers

def fdm(
    weights : np.ndarray,
    forecasts : list[pd.DataFrame]
    ) -> float:

    if not forecasts:
        raise ValueError("At least one forecast is required")

    flattened_forecasts = [forecast.to_numpy(dtype=np.float64).flatten() for forecast in forecasts]

    forecast_df = pd.DataFrame(np.column_stack(flattened_forecasts))

    correlation_matrix = forecast_df.corr().clip(lower=0)

    FDM : float = 1 / np.sqrt(weights.T @ correlation_matrix.to_numpy(dtype=np.float64) @ weights).item()

    return FDM

def regime_scaled_trend(speed : int, risk_object : RiskMeasure, futures : list[Future]) -> pd.DataFrame:
    raw_F : np.ndarray = raw_forecast(speed, futures, risk_object)
    regime_scalar : np.ndarray = regime_scaling(10, risk_object)

    forecast : pd.DataFrame = pd.DataFrame(raw_F * regime_scalar, columns=[future.name for future in futures])
    normalized_forecast : pd.DataFrame = forecast.div(forecast.abs().mean().mean())

    return normalized_forecast

def trend_signals(
        trend_function : Callable[[int, RiskMeasure, list[Future]], pd.DataFrame],
        risk_object : RiskMeasure[Future],
        futures : list[Future],
        speeds : list[int],
        bounds : tuple[int, int] = (-2, 2)
    ) -> pd.DataFrame:

    annualized_std : pd.DataFrame = risk_object.get_var().to_standard_deviation().annualize().to_frame()

    prices : pd.DataFrame = pd.DataFrame()

    for future in futures:
        if prices.empty:
            prices = future.front.get_close().to_frame(future.name)
        else:
            prices = prices.join(future.front.get_close().to_frame(future.name), how="outer")



    forecasts : list[pd.DataFrame] = [trend_function(speed, risk_object, futures).clip(bounds[0], bounds[1]) for speed in speeds]

    weights : np.ndarray = np.ones((len(speeds), 1), dtype=np.float64) / len(forecasts)

    FDM : float = fdm(weights, forecasts)

    average_forecasts = pd.DataFrame(0, index=forecasts[0].index, columns=[future.name for future in futures])

    for forecast in forecasts:
        average_forecasts += forecast

    average_forecasts /= len(forecasts)

    trend_signals = (FDM * average_forecasts).clip(bounds[0], bounds[1])

    return trend_signals

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
