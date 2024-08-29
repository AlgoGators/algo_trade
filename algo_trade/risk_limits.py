import numpy as np
import datetime
import logging
import warnings
from typing import Callable

from algo_trade.risk_logging import LogMessage, LogSubType, LogType
from algo_trade._constants import DAYS_IN_YEAR

def minimum_volatility(max_forecast_ratio : float, IDM : float, tau : float, maximum_leverage : float, instrument_weight : float | np.ndarray, annualized_volatility : float | np.ndarray) -> bool:
    """
    Returns True if the returns for a given instrument meets a minimum level of volatility; else, False
    (works for both single instruments and arrays)

    Parameters:
    ---
        max_forecast_ratio : float
            the max forecast ratio (max forecast / average forecast) ... often 20 / 10 -> 2
        IDM : float
            instrument diversification multiplier
        tau : float
            the target risk for the portfolio
        maximum_leverage : float
            the max acceptable leverage for a given instrument
        instrument_weight : float | np.ndarray
            the weight of the instrument in the portfolio (capital allocated to the instrument / total capital)
            ... often 1/N
        annualized_volatility : float | np.ndarray
            standard deviation of returns for the instrument, in same terms as tau e.g. annualized
    """
    return annualized_volatility >= (max_forecast_ratio * IDM * instrument_weight * tau) / maximum_leverage

def portfolio_multiplier(
        max_portfolio_leverage : float, 
        max_correlation_risk : float, 
        max_portfolio_volatility : float,
        max_portfolio_jump_risk : float) -> Callable:

    def max_leverage(positions_weighted : np.ndarray) -> float:
        """
        Parameters:
        ---
            positions_weighted : np.ndarray
                the notional exposure / position * # positions / capital
                Same as dynamic optimization
        """
        leverage = np.sum(np.abs(positions_weighted))
        return max_portfolio_leverage / leverage

    def correlation_risk(positions_weighted : np.ndarray, annualized_volatility : np.ndarray) -> float:
        """
        Parameters:
        ---
            positions_weighted : np.ndarray
                the notional exposure / position * # positions / capital
                Same as dynamic optimization
            annualized_volatility : np.ndarray
                standard deviation of returns for the instrument, in same terms as tau e.g. annualized
        """
        correlation_risk = np.sum(np.abs(positions_weighted) * annualized_volatility)
        return max_correlation_risk / correlation_risk
    
    def portfolio_risk(positions_weighted : np.ndarray, covariance_matrix : np.ndarray) -> float:
        """
        Parameters:
        ---
            positions_weighted : np.ndarray
                the notional exposure / position * # positions / capital
                Same as dynamic optimization
            covariance_matrix : np.ndarray
                the covariances between the instrument returns
        """
        portfolio_volatility = np.sqrt(positions_weighted @ covariance_matrix @ positions_weighted.T)
        return max_portfolio_volatility / portfolio_volatility

    def jump_risk_multiplier(positions_weighted : np.ndarray, jump_covariance_matrix : np.ndarray) -> float:
        """
        Parameters:
        ---
            maximum_portfolio_jump_risk : float
                the max acceptable jump risk for the portfolio
            positions_weighted : np.ndarray
                the notional exposure / position * # positions / capital
                Same as dynamic optimization
            jumps : np.ndarray
                the jumps in the instrument returns
        """
        jump_risk = np.sqrt(positions_weighted @ jump_covariance_matrix @ positions_weighted.T)
        return max_portfolio_jump_risk / jump_risk

    def fn(
            positions_weighted : np.ndarray,
            covariance_matrix : np.ndarray,
            jump_covariance_matrix : np.ndarray,
            date : datetime.datetime) -> float:
        
        annualized_volatility = np.diag(covariance_matrix) * DAYS_IN_YEAR ** 0.5

        scalars = {
            LogSubType.LEVERAGE_MULTIPLIER : max_leverage(positions_weighted), 
            LogSubType.CORRELATION_MULTIPLIER : correlation_risk(positions_weighted, annualized_volatility),
            LogSubType.VOLATILITY_MULTIPLIER : portfolio_risk(positions_weighted, covariance_matrix),
            LogSubType.JUMP_MULTIPLIER : jump_risk_multiplier(positions_weighted, jump_covariance_matrix)}

        portfolio_scalar = 1
        for key, value in scalars.items():
            if value < 1:
                portfolio_scalar = value
                logging.warning(LogMessage(date, LogType.PORTFOLIO_MULTIPLIER, key, None, value))

        return portfolio_scalar

    return fn

class PositionLimit:
    def max_leverage_position_limit(maximum_leverage : float, capital : float, notional_exposure_per_contract : np.ndarray) -> np.ndarray:
        """
        Returns the lesser of the max leverage limit and the number of contracts to be traded
        (works for both single instruments and arrays)

        Parameters:
        ---
            maximum_leverage : float
                the max acceptable leverage for a given instrument
            capital : float
                the total capital allocated to the portfolio
            notional_exposure_per_contract : np.ndarray
                the notional exposure per contract for the instrument
        """
        return maximum_leverage * capital / notional_exposure_per_contract

    def max_forecast_position_limit(
            maximum_forecast_ratio : float, 
            capital : float, 
            IDM : float, 
            tau : float,
            max_forecast_buffer : float,
            instrument_weight : np.ndarray,
            notional_exposure_per_contract : np.ndarray, 
            annualized_volatility : np.ndarray) -> np.ndarray:
        
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

    def max_pct_of_open_interest_position_limit(max_acceptable_pct_of_open_interest : float, open_interest : np.ndarray) -> np.ndarray:
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

    def volume_limit_positions(volume : np.ndarray, minimum_volume : float, positions : np.ndarray) -> np.ndarray:
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
        contracts : np.ndarray,
        notional_exposure_per_contract : np.ndarray,
        annualized_volatility : np.ndarray,
        instrument_weight : np.ndarray,
        volume : np.ndarray,
        additional_data : tuple[list[str], datetime.datetime],
        maximum_position_leverage : float,
        capital : float,
        IDM : float,
        tau : float,
        maximum_forecast_ratio : float,
        minimum_volume : float,
        max_forecast_buffer : float) -> np.ndarray:
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
        max_leverage_positions = PositionLimit.max_leverage_position_limit(maximum_position_leverage, capital, notional_exposure_per_contract)
        max_forecast_positions = PositionLimit.max_forecast_position_limit(maximum_forecast_ratio, capital, IDM, tau, max_forecast_buffer, instrument_weight, notional_exposure_per_contract, annualized_volatility)
        volume_limited_positions = PositionLimit.volume_limit_positions(volume, minimum_volume, contracts)

        for max_leverage_position, max_forecast_position, volume, contract, instrument_name in zip(max_leverage_positions, max_forecast_positions, volume, contracts, additional_data[0]):
            if contract > max_leverage_position:
                logging.warning(LogMessage(additional_data[1], LogType.POSITION_LIMIT, LogSubType.MAX_LEVERAGE, instrument_name, max_leverage_position))
            if contract > max_forecast_position:
                logging.warning(LogMessage(additional_data[1], LogType.POSITION_LIMIT, LogSubType.MAX_FORECAST, instrument_name, max_forecast_position))
            if volume < minimum_volume:
                logging.warning(LogMessage(additional_data[1], LogType.POSITION_LIMIT, LogSubType.MINIMUM_VOLUME, instrument_name, volume))
            

        return np.minimum(np.minimum(
            max_leverage_positions,
            max_forecast_positions,
            volume_limited_positions), contracts)
