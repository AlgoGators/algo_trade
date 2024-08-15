import datetime
import logging

import numpy as np

import warnings

from ..shared_functions._logging import LogMessage, LogSubType, LogType


def max_leverage_position_limit(maximum_leverage : float, capital : float, notional_exposure_per_contract : float | np.ndarray) -> float | np.ndarray:
    """
    Returns the lesser of the max leverage limit and the number of contracts to be traded
    (works for both single instruments and arrays)

    Parameters:
    ---
        maximum_leverage : float
            the max acceptable leverage for a given instrument
        capital : float
            the total capital allocated to the portfolio
        notional_exposure_per_contract : float | np.ndarray
            the notional exposure per contract for the instrument
    """
    return maximum_leverage * capital / notional_exposure_per_contract

def max_forecast_position_limit(
        maximum_forecast_ratio : float, 
        capital : float, 
        IDM : float, 
        tau : float,
        max_forecast_buffer : float,
        instrument_weight : float | np.ndarray,
        notional_exposure_per_contract : float | np.ndarray, 
        annualized_volatility : float | np.ndarray) -> float | np.ndarray:
    
    """
    Returns the lesser of the max forecast limit and the number of contracts to be traded
    (works for both single instruments and arrays)

    Parameters:
    ---
        maximum_forecast_ratio : float
            the max forecast ratio (max forecast / average forecast) ... often 20 / 10 -> 2
        capital : float
            the total capital allocated to the portfolio
        IDM : float
            instrument diversification multiplier
        tau : float
            the target risk for the portfolio
        instrument_weight : float | np.ndarray
            the weight of the instrument in the portfolio (capital allocated to the instrument / total capital)
            ... often 1/N
        notional_exposure_per_contract : float | np.ndarray
            the notional exposure per contract for the instrument
        annualized_volatility : float | np.ndarray
            standard deviation of returns for the instrument, in same terms as tau e.g. annualized
    """
    return (1 + max_forecast_buffer) * maximum_forecast_ratio * capital * IDM * instrument_weight * tau / notional_exposure_per_contract / annualized_volatility

def max_pct_of_open_interest_position_limit(max_acceptable_pct_of_open_interest : float, open_interest : float | np.ndarray) -> float | np.ndarray:
    """
    Returns the lesser of the max acceptable percentage of open interest and the number of contracts to be traded
    (works for both single instruments and arrays)

    Parameters:
    ---
        max_acceptable_pct_of_open_interest : float
            the max acceptable percentage of open interest for a given instrument
        open_interest : float | np.ndarray
            the open interest for the instrument
    """
    warnings.warn("Switching over to Volume Restriction",
                  DeprecationWarning, 2)
    return max_acceptable_pct_of_open_interest * open_interest

def minimum_volume(volume : float | np.ndarray, minimum_volume : float, positions : float | np.ndarray) -> float | np.ndarray:
    """
    Returns the lesser of the minimum volume and the number of contracts to be traded
    (works for both single instruments and arrays)

    Parameters:
    ---
        volume : float | np.ndarray
            the volume for the instrument
        minimum_volume : float
            minimum volume requirement for any instrument
    """

    volume_mask = np.where(volume < minimum_volume, 0, 1)

    return volume_mask * positions

def position_limit_aggregator(
    maximum_position_leverage : float,
    capital : float,
    IDM : float,
    tau : float,
    maximum_forecast_ratio : float,
    minimum_volume : float,
    max_forecast_buffer : float,
    contracts : float | np.ndarray,
    notional_exposure_per_contract : float | np.ndarray,
    annualized_volatility : float | np.ndarray,
    instrument_weight : float | np.ndarray,
    volumes : float | np.ndarray,
    additional_data : tuple[list[str], datetime.datetime]) -> float | np.ndarray:
    """
    Returns the minimum of the three position limits
    (works for both single instruments and arrays)

    Parameters:
    ---
        maximum_leverage : float
            the max acceptable leverage for a given instrument
        capital : float
            the total capital allocated to the portfolio
        IDM : float
            instrument diversification multiplier
        tau : float
            the target risk for the portfolio
        maximum_forecast_ratio : float
            the max forecast ratio (max forecast / average forecast) ... often 20 / 10 -> 2
        minimum_volume : float
            minimum volume requirement for any instrument
        max_forecast_buffer : float
            the max acceptable buffer for the forecast
        contracts : float | np.ndarray
            the number of contracts to be traded
        notional_exposure_per_contract : float | np.ndarray
            the notional exposure per contract for the instrument
        annualized_volatility : float | np.ndarray
            standard deviation of returns for the instrument, in same terms as tau e.g. annualized
        instrument_weight : float | np.ndarray
            the weight of the instrument in the portfolio (capital allocated to the instrument / total capital)
            ... often 1/N
        volumes : float | np.ndarray
            the volume for the instrument
    """
    if isinstance(contracts, (int, float)):
        return min(
            max_leverage_position_limit(maximum_position_leverage, capital, notional_exposure_per_contract),
            max_forecast_position_limit(maximum_forecast_ratio, capital, IDM, tau, max_forecast_buffer, instrument_weight, notional_exposure_per_contract, annualized_volatility))
    
    max_leverage_positions = max_leverage_position_limit(maximum_position_leverage, capital, notional_exposure_per_contract)
    max_forecast_positions = max_forecast_position_limit(maximum_forecast_ratio, capital, IDM, tau, max_forecast_buffer, instrument_weight, notional_exposure_per_contract, annualized_volatility)
    volume_limited_positions = minimum_volume(volumes, minimum_volume, contracts)

    for max_leverage_position, max_forecast_position, volume, contract, instrument_name in zip(max_leverage_positions, max_forecast_positions, volumes, contracts, additional_data[0]):
        if contract > max_leverage_position:
            logging.warning(LogMessage(additional_data[1], LogType.POSITION_LIMIT, LogSubType.MAX_LEVERAGE, instrument_name, max_leverage_position))
        if contract > max_forecast_position:
            logging.warning(LogMessage(additional_data[1], LogType.POSITION_LIMIT, LogSubType.MAX_FORECAST, instrument_name, max_forecast_position))
        if volume < minimum_volume:
            logging.warning(LogMessage(additional_data[1], LogType.POSITION_LIMIT, LogSubType.MINIMUM_VOLUME, instrument_name, minimum_volume))
        

    return np.minimum(np.minimum(
        max_leverage_positions,
        max_forecast_positions,
        volume_limited_positions), contracts)

