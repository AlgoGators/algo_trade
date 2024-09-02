import pandas as pd
import numpy as np

from algo_trade.instrument import Future
from algo_trade.risk_measures import RiskMeasure

### Strategy Rules

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

def risk_parity(risk_object: RiskMeasure) -> pd.DataFrame:
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
                .get_backadjusted()
                .ewm(span=t1, min_periods=2, adjust=False)
                .mean()
                - instrument.front
                .get_backadjusted()
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

def IDM(instruments: list[Future]) -> pd.DataFrame:
    raise NotImplementedError("IDM not implemented")
