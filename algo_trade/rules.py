import pandas as pd # type: ignore
import numpy as np

from typing import TypeVar

from algo_trade.instrument import Future, Instrument
from algo_trade.risk_measures import RiskMeasure
from algo_trade._constants import DAYS_IN_YEAR

### Strategy Rules

T = TypeVar('T', bound=Instrument)

def capital_scaling(instruments: list[Future], capital: float) -> pd.DataFrame:
    df = pd.DataFrame()
    instrument: Future
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
                how="outer",
            )

    df.ffill(inplace=True)

    capital_weighting = capital / df

    return capital_weighting

def risk_parity(risk_object: RiskMeasure[T]) -> pd.DataFrame:
    if risk_object.tau is None:
        raise ValueError("Risk Measure object must have a tau value")

    return risk_object.tau / risk_object.get_var().to_standard_deviation().annualize()

def equal_weight(instruments: list[Future]) -> pd.DataFrame:
    df = pd.DataFrame()
    instrument: Future
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
                how="outer",
            )

    df.ffill(inplace=True)

    not_null_mask = df.notnull()
    weight_mask = 1 / df.notnull().sum(axis=1).astype(int)

    weights = not_null_mask.mul(weight_mask, axis=0)

    return weights

def trend_signals(instruments: list[Future], risk_object : RiskMeasure) -> pd.DataFrame:
    forecasts: list[pd.Series] = []
    instrument: Future
    for instrument in instruments:
        trend = pd.DataFrame()

        crossovers = [(2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256)]

        # Calculate the exponential moving averages crossovers and store them in the trend dataframe for t1, t2 in crossovers: trend[f"{t1}-{t2}"] = data["Close"].ewm(span=t1, min_periods=2).mean() - data["Close"].ewm(span=t2, min_periods=2).mean()
        for t1, t2 in crossovers:
            trend[f"{t1}-{t2}"] = (
                instrument.front
                .backadjusted
                .ewm(span=t1, min_periods=2, adjust=False)
                .mean()
                - instrument.front
                .backadjusted
                .ewm(span=t2, min_periods=2, adjust=False)
                .mean()
            )

        # Calculate the risk adjusted forecasts
        for t1, t2 in crossovers:
            trend[f"{t1}-{t2}"] /= (
                risk_object
                .get_var()
                .to_standard_deviation()
                .annualize()
                [instrument.get_symbol()]
                * instrument.front.close
            )

        # Scale the crossovers by the absolute mean of all previous crossovers
        scalar_dict = {}
        for t1, t2 in crossovers:
            scalar_dict[t1] = 10 / trend[f"{t1}-{t2}"].abs().mean()

        for t1, t2 in crossovers:
            trend[f"{t1}-{t2}"] = trend[f"{t1}-{t2}"] * scalar_dict[t1]

        # Clip the scaled crossovers to -20, 20
        for t1, t2 in crossovers:
            trend[f"{t1}-{t2}"] = trend[f"{t1}-{t2}"].clip(-20, 20)

        trend.Forecast = 0.0

        n = len(crossovers)
        weights = {64: 1 / n, 32: 1 / n, 16: 1 / n, 8: 1 / n, 4: 1 / n, 2: 1 / n}

        for t1, t2 in crossovers:
            trend.Forecast += trend[f"{t1}-{t2}"] * weights[t1]

        fdm = 1.35
        trend.Forecast = trend.Forecast * fdm

        # Clip the final forecast to -20, 20
        trend.Forecast = trend.Forecast.clip(-20, 20)

        forecasts.append((trend.Forecast / 10).rename(instrument.name))

    df = pd.DataFrame()
    for series in forecasts:
        if df.empty:
            df = series.to_frame()
        else:
            df = df.join(series.to_frame(), how="outer")

    return df

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
