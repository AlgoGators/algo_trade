import pandas as pd
import numpy as np
import datetime
import logging
from typing import Callable
from functools import reduce

from algo_trade.portfolio import Portfolio
from algo_trade.instrument import Future
from algo_trade.risk_logging import CsvFormatter

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler('log.csv', mode='w'),
        logging.StreamHandler()])

logger = logging.getLogger(__name__)
logging.root.handlers[0].setFormatter(CsvFormatter())


def reindex(dfs : tuple[pd.DataFrame]) -> tuple[pd.DataFrame]:
    dfs = [df.set_index(pd.to_datetime(df.index)).astype(np.float64).ffill().dropna() for df in dfs]
    intersection_index = reduce(lambda x, y: x.intersection(y), [df.index for df in dfs if isinstance(df, pd.DataFrame)])

    dfs = [df.loc[intersection_index] for df in dfs]

    return dfs

def get_cost_penalty(x_weighted : np.ndarray, y_weighted : np.ndarray, weighted_cost_per_contract : np.ndarray, cost_penalty_scalar : int) -> float:
    """Finds the trading cost to go from x to y, given the weighted cost per contract and the cost penalty scalar"""

    #* Should never activate but just in case
    # x_weighted = np.nan_to_num(np.asarray(x_weighted, dtype=np.float64))
    # y_weighted = np.nan_to_num(np.asarray(y_weighted, dtype=np.float64))
    # weighted_cost_per_contract = np.nan_to_num(np.asarray(weighted_cost_per_contract, dtype=np.float64))

    trading_cost = np.abs(x_weighted - y_weighted) * weighted_cost_per_contract

    return np.sum(trading_cost) * cost_penalty_scalar

def get_portfolio_tracking_error_standard_deviation(x_weighted : np.ndarray, y_weighted : np.ndarray, covariance_matrix : np.ndarray, cost_penalty : float = 0.0) -> float:
    if np.isnan(x_weighted).any() or np.isnan(y_weighted).any() or np.isnan(covariance_matrix).any():
        raise ValueError("Input contains NaN values")
    
    tracking_errors = x_weighted - y_weighted

    dot_product = np.dot(np.dot(tracking_errors, covariance_matrix), tracking_errors)

    #* deal with negative radicand (really, REALLY shouldn't happen)
    #? maybe its a good weight set but for now, it's probably safer this way
    if dot_product < 0:
        return 1.0
    
    return np.sqrt(dot_product) + cost_penalty

def covariance_row_to_matrix(row : np.ndarray) -> np.ndarray:
    num_instruments = int(np.sqrt(2 * len(row)))
    matrix = np.zeros((num_instruments, num_instruments))

    idx = 0
    for i in range(num_instruments):
        for j in range(i, num_instruments):
            matrix[i, j] = matrix[j, i] = row[idx]
            idx += 1

    return matrix

def round_multiple(x : np.ndarray, multiple : np.ndarray) -> np.ndarray:
    return np.round(x / multiple) * multiple

def buffer_weights(optimized : np.ndarray, held : np.ndarray, weights : np.ndarray, covariance_matrix : np.ndarray, tau : float, asymmetric_risk_buffer : float):
    tracking_error = get_portfolio_tracking_error_standard_deviation(optimized, held, covariance_matrix)

    tracking_error_buffer = tau * asymmetric_risk_buffer

    # If the tracking error is less than the buffer, we don't need to do anything
    if tracking_error < tracking_error_buffer or tracking_error == 0:
        return held

    adjustment_factor = max((tracking_error - tracking_error_buffer) / tracking_error, 0.0)
    if np.isnan(adjustment_factor):
        raise ValueError(f"Invalid value encountered in scalar divide: tracking_error={tracking_error}")

    required_trades = round_multiple((optimized - held) * adjustment_factor, weights)

    return held + required_trades

# Might be worth framing this similar to scipy.minimize function in terms of argument names (or quite frankly, maybe just use scipy.minimize)
def greedy_algorithm(ideal : np.ndarray, x0 : np.ndarray, weighted_costs_per_contract : np.ndarray, held : np.ndarray, weights_per_contract : np.ndarray, covariance_matrix : np.ndarray, cost_penalty_scalar : int) -> np.ndarray:
    if ideal.ndim != 1 or ideal.shape != x0.shape != held.shape != weights_per_contract.shape != weighted_costs_per_contract.shape:
        raise ValueError("Input shapes do not match")
    if covariance_matrix.ndim != 2 or covariance_matrix.shape[0] != covariance_matrix.shape[1] or len(ideal) != covariance_matrix.shape[0]:
        raise ValueError("Invalid covariance matrix (should be [N x N])")
    
    proposed_solution = x0.copy()
    cost_penalty = get_cost_penalty(held, proposed_solution, weighted_costs_per_contract, cost_penalty_scalar)
    tracking_error = get_portfolio_tracking_error_standard_deviation(ideal, proposed_solution, covariance_matrix, cost_penalty)
    best_tracking_error = tracking_error
    iteration_limit = 1000
    iteration = 0

    while iteration <= iteration_limit:
        previous_solution = proposed_solution.copy()
        best_IDX = None

        for idx in range(len(proposed_solution)):
            temp_solution = previous_solution.copy()

            if temp_solution[idx] < ideal[idx]:
                temp_solution[idx] += weights_per_contract[idx]
            else:
                temp_solution[idx] -= weights_per_contract[idx]

            cost_penalty = get_cost_penalty(held, temp_solution, weighted_costs_per_contract, cost_penalty_scalar)
            tracking_error = get_portfolio_tracking_error_standard_deviation(ideal, temp_solution, covariance_matrix, cost_penalty)

            if tracking_error <= best_tracking_error:
                best_tracking_error = tracking_error
                best_IDX = idx

        if best_IDX is None:
            break

        if proposed_solution[best_IDX] <= ideal[best_IDX]:
            proposed_solution[best_IDX] += weights_per_contract[best_IDX]
        else:
            proposed_solution[best_IDX] -= weights_per_contract[best_IDX]
        
        iteration += 1

    return proposed_solution

def single_day_optimization(
        held_positions_one_day : np.ndarray,
        ideal_positions_one_day : np.ndarray,
        costs_per_contract_one_day : np.ndarray,
        weight_per_contract_one_day : np.ndarray,
        instrument_weight_one_day : np.ndarray,
        notional_exposure_per_contract_one_day : np.ndarray,
        covariances_one_day : np.ndarray,
        jump_covariances_one_day : np.ndarray,
        volume_one_day : np.ndarray,
        tau : float,
        capital : float,
        asymmetric_risk_buffer : float,
        cost_penalty_scalar : int,
        additional_data : tuple[list[str], list[datetime.datetime]],
        optimization : bool,
        position_limit_fn : Callable,
        portfolio_multiplier_fn : Callable) -> np.ndarray:

    covariance_matrix_one_day : np.ndarray = covariance_row_to_matrix(covariances_one_day)
    jump_covariance_matrix_one_day : np.ndarray = covariance_row_to_matrix(jump_covariances_one_day)

    ideal_positions_weighted = ideal_positions_one_day * weight_per_contract_one_day
    held_positions_weighted = held_positions_one_day * weight_per_contract_one_day
    costs_per_contract_weighted = costs_per_contract_one_day / capital / weight_per_contract_one_day

    x0 : np.ndarray = held_positions_weighted

    if optimization:
        optimized_weights_one_day = greedy_algorithm(ideal_positions_weighted, x0, costs_per_contract_weighted, held_positions_weighted, weight_per_contract_one_day, covariance_matrix_one_day, cost_penalty_scalar)

        buffered_weights = buffer_weights(
            optimized_weights_one_day, held_positions_weighted, weight_per_contract_one_day,
            covariance_matrix_one_day, tau, asymmetric_risk_buffer)

        optimized_positions_one_day = buffered_weights / weight_per_contract_one_day
    else:
        optimized_positions_one_day = ideal_positions_weighted / weight_per_contract_one_day

    risk_limited_positions = np.minimum(
        position_limit_fn(
            capital, notional_exposure_per_contract_one_day, instrument_weight_one_day,
            covariance_matrix_one_day, volume_one_day, additional_data
        ),
        optimized_positions_one_day
    )

    risk_limited_positions_weighted = risk_limited_positions * weight_per_contract_one_day

    portfolio_risk_limited_positions = risk_limited_positions * portfolio_multiplier_fn(
        risk_limited_positions_weighted, covariance_matrix_one_day, 
        jump_covariance_matrix_one_day, date=additional_data[1])

    return round_multiple(portfolio_risk_limited_positions, 1) if optimization else portfolio_risk_limited_positions

def dyn_opt(
        portfolio : Portfolio[Future],
        instrument_weights : pd.DataFrame,
        cost_per_contract : float,
        asymmetric_risk_buffer : float,
        cost_penalty_scalar : float,
        position_limit_fn : Callable,
        portfolio_multiplier_fn : Callable) -> Portfolio[Future]:
    
    unadj_prices = pd.concat([instrument.front.close.rename(instrument.name) for instrument in portfolio.instruments], axis=1)
    covariances = portfolio.risk_object.get_cov()
    jump_covariances : pd.DataFrame = portfolio.risk_object.get_jump_cov(0.95, 100)
    volume = pd.concat([instrument.front.volume.rename(instrument.name) for instrument in portfolio.instruments], axis=1)
    
    costs_per_contract = pd.DataFrame(index=portfolio.positions.index, columns=portfolio.positions.columns).astype(np.float64).fillna(cost_per_contract)
    
    portfolio.positions, costs_per_contract, covariances, jump_covariances, volume, unadj_prices,instrument_weights = reindex((portfolio.positions, costs_per_contract, covariances, jump_covariances, volume, unadj_prices, instrument_weights))

    notional_exposure_per_contract = unadj_prices * portfolio.multipliers.iloc[0]
    weight_per_contract = notional_exposure_per_contract / portfolio.capital

    optimized_positions = pd.DataFrame(index=portfolio.positions.index, columns=portfolio.positions.columns).astype(np.float64)

    position_matrix = portfolio.positions.values
    cost_matrix = costs_per_contract.values
    contract_weight_matrix = weight_per_contract.values
    exposure_matrix = notional_exposure_per_contract.values
    volume_matrix = volume.values
    covariance_matrix = covariances.values
    jump_covariance_matrix = jump_covariances.values
    instrument_weight_matrix = instrument_weights.values

    for n, date in enumerate(portfolio.positions.index):
        held_positions_vector = np.zeros(len(portfolio.instruments))

        if n != 0:
            current_date_IDX = portfolio.positions.index.get_loc(date)
            held_positions_vector = optimized_positions.iloc[current_date_IDX - 1].values

        optimized_positions.iloc[n] = single_day_optimization(
            held_positions_vector,
            position_matrix[n],
            cost_matrix[n],
            contract_weight_matrix[n],
            instrument_weight_matrix[n],
            exposure_matrix[n],
            covariance_matrix[n],
            jump_covariance_matrix[n],
            volume_matrix[n],
            portfolio.risk_object.tau,
            portfolio.capital,
            asymmetric_risk_buffer,
            cost_penalty_scalar,
            (portfolio.instruments, date),
            True,
            position_limit_fn,
            portfolio_multiplier_fn
        )

    return optimized_positions
