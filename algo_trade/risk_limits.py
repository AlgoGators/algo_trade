import numpy as np
import datetime
import logging
import warnings

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

class PortfolioMultiplier:
    def max_leverage_portfolio_multiplier(maximum_portfolio_leverage : float, positions_weighted : np.ndarray) -> float:
        """
        Returns the positions scaled by the max leverage limit

        Parameters:
        ---
            maximum_portfolio_leverage : float
                the max acceptable leverage for the portfolio
            positions_weighted : np.ndarray
                the notional exposure / position * # positions / capital
                Same as dynamic optimization
        """
        leverage = np.sum(np.abs(positions_weighted))
        scalar = np.minimum(maximum_portfolio_leverage / leverage, 1)

        return scalar

    def correlation_risk_portfolio_multiplier(maximum_portfolio_correlation_risk : float, positions_weighted : np.ndarray, annualized_volatility : np.ndarray) -> float:
        """
        Returns the positions scaled by the correlation risk limit

        Parameters:
        ---
            positions_weighted : np.ndarray
                the notional exposure / position * # positions / capital
                Same as dynamic optimization
            annualized_volatility : np.ndarray
                standard deviation of returns for the instrument, in same terms as tau e.g. annualized
        """
        correlation_risk = np.sum(np.abs(positions_weighted) * annualized_volatility)
        scalar = np.minimum(1, maximum_portfolio_correlation_risk / correlation_risk)

        return scalar

    def portfolio_risk_multiplier(
            maximum_portfolio_volatility : float, 
            positions_weighted : np.ndarray, 
            covariance_matrix : np.ndarray) -> float:
        """
        Returns the positions scaled by the portfolio volatility limit

        Parameters:
        ---
            maximum_portfolio_volatility : float
                the max acceptable volatility for the portfolio
            positions_weighted : np.ndarray
                the notional exposure / position * # positions / capital
                Same as dynamic optimization
            covariance_matrix : np.ndarray
                the covariances between the instrument returns
        """
        portfolio_volatility = np.sqrt(positions_weighted @ covariance_matrix @ positions_weighted.T)
        scalar = np.minimum(1, maximum_portfolio_volatility / portfolio_volatility)

        return scalar

    def jump_risk_multiplier(maximum_portfolio_jump_risk : float, positions_weighted : np.ndarray, jump_covariance_matrix) -> float:
        """
        Returns the positions scaled by the jump risk limit

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
        scalar = np.minimum(1, maximum_portfolio_jump_risk / jump_risk)

        return scalar

    def portfolio_risk_aggregator(
            positions : np.ndarray,
            positions_weighted : np.ndarray, 
            covariance_matrix : np.ndarray, 
            jump_covariance_matrix : np.ndarray,
            maximum_portfolio_leverage : float,
            maximum_correlation_risk : float,
            maximum_portfolio_risk : float,
            maximum_jump_risk : float,
            date : datetime.datetime) -> np.ndarray:

        annualized_volatilities = np.diag(covariance_matrix) * DAYS_IN_YEAR ** 0.5

        leverage_multiplier = PortfolioMultiplier.max_leverage_portfolio_multiplier(maximum_portfolio_leverage, positions_weighted)
        correlation_multiplier = PortfolioMultiplier.correlation_risk_portfolio_multiplier(maximum_correlation_risk, positions_weighted, annualized_volatilities)
        volatility_multiplier = PortfolioMultiplier.portfolio_risk_multiplier(maximum_portfolio_risk, positions_weighted, covariance_matrix)
        jump_multiplier = PortfolioMultiplier.jump_risk_multiplier(maximum_jump_risk, positions_weighted, jump_covariance_matrix)

        if leverage_multiplier < 1:
            logging.warning(LogMessage(date, LogType.PORTFOLIO_MULTIPLIER, LogSubType.LEVERAGE_MULTIPLIER, None, leverage_multiplier))
        if correlation_multiplier < 1:
            logging.warning(LogMessage(date, LogType.PORTFOLIO_MULTIPLIER, LogSubType.CORRELATION_MULTIPLIER, None, correlation_multiplier))
        if volatility_multiplier < 1:
            logging.warning(LogMessage(date, LogType.PORTFOLIO_MULTIPLIER, LogSubType.VOLATILITY_MULTIPLIER, None, volatility_multiplier))
        if jump_multiplier < 1:
            logging.warning(LogMessage(date, LogType.PORTFOLIO_MULTIPLIER, LogSubType.JUMP_MULTIPLIER, None, jump_multiplier))

        return positions * min(leverage_multiplier, correlation_multiplier, volatility_multiplier, jump_multiplier)

class PositionLimit:
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

    def volume_limit_positions(volume : float | np.ndarray, minimum_volume : float, positions : float | np.ndarray) -> float | np.ndarray:
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
                PositionLimit.max_leverage_position_limit(maximum_position_leverage, capital, notional_exposure_per_contract),
                PositionLimit.max_forecast_position_limit(maximum_forecast_ratio, capital, IDM, tau, max_forecast_buffer, instrument_weight, notional_exposure_per_contract, annualized_volatility))
        
        max_leverage_positions = PositionLimit.max_leverage_position_limit(maximum_position_leverage, capital, notional_exposure_per_contract)
        max_forecast_positions = PositionLimit.max_forecast_position_limit(maximum_forecast_ratio, capital, IDM, tau, max_forecast_buffer, instrument_weight, notional_exposure_per_contract, annualized_volatility)
        volume_limited_positions = PositionLimit.volume_limit_positions(volumes, minimum_volume, contracts)

        for max_leverage_position, max_forecast_position, volume, contract, instrument_name in zip(max_leverage_positions, max_forecast_positions, volumes, contracts, additional_data[0]):
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