import numpy as np
import pandas as pd


def daily_variance_to_annualized_volatility(daily_variance : float | np.ndarray) -> float | np.ndarray:
    return (daily_variance * 256) ** 0.5

def get_jump_covariances(covariances : pd.DataFrame, percentile : float, window : int) -> pd.DataFrame:
    dates = covariances.index

    jump_covariances = pd.DataFrame(index=dates, columns=covariances.columns)

    for i in range(len(dates)):
        if i < window:
            continue

        window_covariances = covariances.iloc[i-window:i]
        jump_covariances.iloc[i] = window_covariances.quantile(percentile)

    return jump_covariances