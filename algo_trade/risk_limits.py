import numpy as np
import numpy.typing as npt
import datetime
import logging
from typing import Callable

from algo_trade.risk_logging import LogMessage, LogSubType, LogType
from algo_trade._constants import DAYS_IN_YEAR

def minimum_volatility(
        max_forecast_ratio : float,
        IDM : float,
        tau : float,
        maximum_leverage : float,
        instrument_weight : float | npt.NDArray[np.float64],
        annualized_volatility : float | npt.NDArray[np.float64]
    ) -> bool | npt.NDArray[np.bool_]:
    """
    Returns True if the returns for a given instrument meets a minimum level of volatility;
    else, False
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
        max_portfolio_leverage : np.float64, 
        max_correlation_risk : np.float64, 
        max_portfolio_volatility : np.float64,
        max_portfolio_jump_risk : np.float64
    ) -> Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            datetime.datetime
        ],
        np.float64]:

    def max_leverage(
            positions_weighted : npt.NDArray[np.float64]
        ) -> np.float64:
        """
        Parameters:
        ---
            positions_weighted : np.ndarray
                the notional exposure / position * # positions / capital
                Same as dynamic optimization
        """
        leverage : np.float64 = np.sum(np.abs(positions_weighted))
        if leverage == 0:
            return np.float64(1)

        if max_portfolio_leverage / leverage < 1:
            return max_portfolio_leverage / leverage

        return np.float64(1)

    def correlation_risk(
            positions_weighted : npt.NDArray[np.float64],
            annualized_volatility : npt.NDArray[np.float64]
        ) -> np.float64:
        """
        Parameters:
        ---
            positions_weighted : np.ndarray (dollars allocated to each instrument)
                the notional exposure * # positions / capital
                Same as dynamic optimization
            annualized_volatility : np.ndarray
                standard deviation of returns for the instrument, in same terms as tau e.g. annualized
        """
        correlation_risk : np.float64 = np.sum(np.abs(positions_weighted) * annualized_volatility)
        if correlation_risk == 0:
            return np.float64(1)

        if max_correlation_risk / correlation_risk < 1:
            return max_correlation_risk / correlation_risk

        return np.float64(1)
    
    def portfolio_risk(
            positions_weighted : npt.NDArray[np.float64],
            covariance_matrix : npt.NDArray[np.float64]
        ) -> np.float64:
        """
        Parameters:
        ---
            positions_weighted : np.ndarray
                the notional exposure / position * # positions / capital
                Same as dynamic optimization
            covariance_matrix : np.ndarray
                the covariances between the instrument returns
        """
        portfolio_volatility : np.float64 = np.sqrt(positions_weighted @ covariance_matrix @ positions_weighted.T)
        if portfolio_volatility == 0:
            return np.float64(1)

        if max_portfolio_volatility / portfolio_volatility < 1:
            return max_portfolio_volatility / portfolio_volatility

        return np.float64(1)

    def jump_risk_multiplier(
            positions_weighted : npt.NDArray[np.float64],
            jump_covariance_matrix : npt.NDArray[np.float64]
        ) -> np.float64:
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
        jump_risk : np.float64 = np.sqrt(
            positions_weighted @ jump_covariance_matrix @ positions_weighted.T)
        if jump_risk == 0:
            return np.float64(1)

        if max_portfolio_jump_risk / jump_risk < 1:
            return max_portfolio_jump_risk / jump_risk

        return np.float64(1)

    def fn(
            positions_weighted : npt.NDArray[np.float64],
            covariance_matrix : npt.NDArray[np.float64],
            jump_covariance_matrix : npt.NDArray[np.float64],
            date : datetime.datetime
        ) -> np.float64:

        annualized_volatility = np.diag(covariance_matrix) * DAYS_IN_YEAR ** 0.5

        scalars = {
            LogSubType.LEVERAGE_MULTIPLIER : max_leverage(positions_weighted), 
            LogSubType.CORRELATION_MULTIPLIER : correlation_risk(positions_weighted, annualized_volatility),
            LogSubType.VOLATILITY_MULTIPLIER : portfolio_risk(positions_weighted, covariance_matrix),
            LogSubType.JUMP_MULTIPLIER : jump_risk_multiplier(positions_weighted, jump_covariance_matrix)}

        portfolio_scalar = np.float64(1)
        for key, value in scalars.items():
            if value < 1:
                portfolio_scalar = value
                logging.warning(
                    LogMessage(
                        DATE=date,
                        TYPE=LogType.PORTFOLIO_MULTIPLIER,
                        SUBTYPE=key,
                        ADDITIONAL_INFO=str(value)))

        return portfolio_scalar

    return fn

def position_limit(
        max_leverage_ratio : int,
        minimum_volume : int,
        max_forecast_ratio : float,
        max_forecast_buffer : float,
        IDM : float,
        tau : float
    ) -> Callable[
        [
            float,
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            tuple[list[str], datetime.datetime]
        ],
        npt.NDArray[np.float64]]:

    def max_leverage(
            capital : float,
            notional_exposure_per_contract : npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
        """
        Parameters:
        ---
            maximum_leverage : float
                the max acceptable leverage for a given instrument
            capital : float
                the total capital allocated to the portfolio
            notional_exposure_per_contract : np.ndarray
                the notional exposure per contract for the instrument
        """
        return max_leverage_ratio * capital / notional_exposure_per_contract

    def max_forecast(
            capital : float,
            notional_exposure_per_contract : npt.NDArray[np.float64],
            instrument_weight : npt.NDArray[np.float64],
            annualized_volatility : npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
        """
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
        return (1 + max_forecast_buffer) * max_forecast_ratio * capital * IDM * instrument_weight * tau / notional_exposure_per_contract / annualized_volatility

    def min_volume(
            volume : npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
        """
        Parameters:
        ---
            volume : float | np.ndarray
                the volume for the instrument
            minimum_volume : float
                minimum volume requirement for any instrument
        """
        volume_mask = np.where(volume < minimum_volume, 0, 1)

        return volume_mask

    def fn(
            capital : float,
            positions : npt.NDArray[np.float64],
            notional_exposure_per_contract : npt.NDArray[np.float64],
            instrument_weight : npt.NDArray[np.float64],
            covariance_matrix : npt.NDArray[np.float64],
            volume : npt.NDArray[np.float64],
            additional_data : tuple[list[str], datetime.datetime]
        ) -> npt.NDArray[np.float64]:

        annualized_volatility = np.diag(covariance_matrix) * DAYS_IN_YEAR ** 0.5

        positions_at_maximum_leverage = abs(max_leverage(capital, notional_exposure_per_contract))
        positions_at_maximum_forecast = abs(max_forecast(capital, notional_exposure_per_contract, instrument_weight, annualized_volatility))
        volume_mask = min_volume(volume)

        max_positions =  volume_mask * np.minimum(positions_at_maximum_leverage, positions_at_maximum_forecast)

        for idx, _ in enumerate(volume_mask):
            if volume_mask[idx] == 0:
                logging.warning(
                    LogMessage(
                        additional_data[1],
                        LogType.POSITION_LIMIT,
                        LogSubType.MINIMUM_VOLUME,
                        additional_data[0][idx],
                        str(0)))

        for position_at_maximum_leverage, position in zip(positions_at_maximum_leverage, positions):
            if abs(position) > position_at_maximum_leverage:
                logging.warning(
                    LogMessage(
                        additional_data[1],
                        LogType.POSITION_LIMIT,
                        LogSubType.MAX_LEVERAGE,
                        additional_data[0][np.where(positions == position)[0][0]],
                        position_at_maximum_leverage))
         
        for position_at_maximum_forecast, position in zip(positions_at_maximum_forecast, positions):
            if abs(position) > position_at_maximum_forecast:
                logging.warning(
                    LogMessage(
                        additional_data[1],
                        LogType.POSITION_LIMIT,
                        LogSubType.MAX_FORECAST,
                        additional_data[0][np.where(positions == position)[0][0]],
                        position_at_maximum_forecast))
     
        sign_map : npt.NDArray[np.float64] = np.sign(positions)

        minimum_position = np.minimum(abs(positions), max_positions) * sign_map

        return minimum_position

    return fn
