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

def raw_forecast(
    n : int,
    future : Future
    ) -> npt.NDArray[np.float64]:
    
    prices = np.array(future.front.backadjusted)
    N = len(prices)

    weight_matrix_n : np.ndarray = np.zeros((N, N))
    weight_matrix_4n : np.ndarray = np.zeros((N, N))

    for i in range(N):
        weight_matrix_n[i, i] = (2 / (n + 1)) ** i * (1 - (2 / (n + 1))) 
        weight_matrix_4n[i, i] = (2 / (4 * n + 1)) ** i * (1 - (2 / (4 * n + 1))) 
    
    y_n = prices @ weight_matrix_n
    y_4n = prices @ weight_matrix_4n

    return np.array(y_n - y_4n)

# def scaled_forecast(
#         crossover : tuple[int, int],
#         instruments : list[Future],
#         risk_object : RiskMeasure[Future]
#     ) -> pd.DataFrame:

#     trend = pd.DataFrame(dtype=float)
#     for instrument in instruments:
#         trend[instrument.symbol] = (
#             instrument.front
#             .backadjusted
#             .ewm(span=crossover[0], min_periods=crossover[0], adjust=False)
#             .mean()
#             - instrument.front
#             .backadjusted
#             .ewm(span=crossover[1], min_periods=crossover[1], adjust=False)
#             .mean()
#         )

#     std : StandardDeviation = risk_object.get_var().to_standard_deviation()
#     std.annualize(inplace=True)

#     for instrument in instruments:
#         trend[instrument.symbol] /= (
#             std[instrument.symbol]
#             * instrument.front.close
#         )
    
#     forecast_scalar = 10 / trend.abs().mean().mean()

#     trend *= forecast_scalar

# def trend_signals(
#         instruments: list[Future],
#         risk_object : RiskMeasure[Future]
#     ) -> pd.DataFrame:

#     forecasts: list[pd.Series] = []
#     for instrument in instruments:
#         trend = pd.DataFrame(dtype=float)

#         crossovers = [(2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256)]

#         # Calculate the exponential moving averages crossovers and store them in the trend dataframe for t1, t2 in crossovers: trend[f"{t1}-{t2}"] = data["Close"].ewm(span=t1, min_periods=2).mean() - data["Close"].ewm(span=t2, min_periods=2).mean()
#         for t1, t2 in crossovers:
#             trend[f"{t1}-{t2}"] = (
#                 instrument.front
#                 .backadjusted
#                 .ewm(span=t1, min_periods=2, adjust=False)
#                 .mean()
#                 - instrument.front
#                 .backadjusted
#                 .ewm(span=t2, min_periods=2, adjust=False)
#                 .mean()
#             )

#         std = risk_object.get_var().to_standard_deviation()
#         std.annualize(inplace=True)

#         # Calculate the risk adjusted forecasts
#         for t1, t2 in crossovers:
#             trend[f"{t1}-{t2}"] /= (
#                 std[instrument.get_symbol()]
#                 * instrument.front.close
#             )

#         # Scale the crossovers by the absolute mean of all previous crossovers
#         scalar_dict = {}
#         for t1, t2 in crossovers:
#             scalar_dict[t1] = 10 / trend[f"{t1}-{t2}"].abs().mean()

#         for t1, t2 in crossovers:
#             trend[f"{t1}-{t2}"] = trend[f"{t1}-{t2}"] * scalar_dict[t1]

#         # Clip the scaled crossovers to -20, 20
#         for t1, t2 in crossovers:
#             trend[f"{t1}-{t2}"] = trend[f"{t1}-{t2}"].clip(-20, 20)

#         trend.Forecast = 0.0

#         n = len(crossovers)
#         weights = {64: 1 / n, 32: 1 / n, 16: 1 / n, 8: 1 / n, 4: 1 / n, 2: 1 / n}

#         for t1, t2 in crossovers:
#             trend.Forecast += trend[f"{t1}-{t2}"] * weights[t1]

#         fdm = 1.35
#         trend.Forecast = trend.Forecast * fdm

#         # # Clip the final forecast to -20, 20
#         # trend.Forecast = trend.Forecast.clip(-20, 20)

#         forecasts.append((trend.Forecast / 10).rename(instrument.name))

#     df = pd.DataFrame()
#     for series in forecasts:
#         if df.empty:
#             df = series.to_frame()
#         else:
#             df = df.join(series.to_frame(), how="outer")

#     return df

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

def volatility_regime_scaling(risk_object : RiskMeasure) -> pd.DataFrame:
    rolling_vol_means = (
        risk_object
        .get_var()
        .to_standard_deviation()
        .to_frame()
        .rolling(window=DAYS_IN_YEAR*10, min_periods=DAYS_IN_YEAR)
        .mean())
    
    relative_vol = (
        risk_object
        .get_var()
        .to_standard_deviation()
        .to_frame()
        .div(rolling_vol_means)
    )

    percentiles = relative_vol.apply(lambda x : x.rank() / x.count(), axis=0)

    percentiles = percentiles.where(pd.notnull(percentiles), 0)

    percentiles = percentiles.astype(float)

    multipliers = (2 - 1.5 * percentiles).ewm(span=10).mean()

    return multipliers

df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [6, 5, 2],
    'C': [7, 5, 1]
})

df = df.rolling(window=3, min_periods=1).mean()

# rolling_vol_means = x.rolling(window=3, min_periods=1).mean()

# relative_vol = x.div(rolling_vol_means)

percentiles = df.apply(lambda x : x.rank() / x.count(), axis=0)

percentiles = percentiles.where(pd.notnull(percentiles), 1.0)

percentiles = percentiles.astype(float)

print(percentiles)