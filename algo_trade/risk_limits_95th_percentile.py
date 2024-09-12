import numpy as np
from functools import partial

from algo_trade.portfolio import Portfolio
from algo_trade.strategy import Strategy, FutureDataFetcher
from algo_trade.instrument import Instrument, Future
from algo_trade.pnl import PnL
from algo_trade.rules import capital_scaling, risk_parity, equal_weight, trend_signals, IDM
from algo_trade.risk_measures import GARCH, RiskMeasure
from algo_trade.dyn_opt import dyn_opt
from algo_trade.risk_limits import portfolio_multiplier, position_limit

class TrendFollowing(Strategy[Future]):
    def __init__(self, instruments: list[Future], risk_object: RiskMeasure, capital: float):
        super().__init__(capital=capital)
        self.instruments: list[Future] = instruments
        self.risk_object = risk_object
        self.rules = [
            partial(risk_parity, risk_object=self.risk_object),
            partial(trend_signals, instruments=instruments, risk_object=self.risk_object),
            partial(equal_weight, instruments=instruments),
            partial(capital_scaling, instruments=instruments, capital=capital),
            partial(IDM, risk_object=self.risk_object)
        ]
        self.scalars = []
        self.fetch_data()  # Fetch the data for the instruments

    def fetch_data(self) -> None:
        """
        The Fetch data method for the Trend Following strategy is requires the following instrument specific data:
        1. Prices(Open, High, Low, Close, Volume)
        2. Backadjusted Prices (Close)
        """
        # Load the front calendar contract data with a daily aggregation
        FutureDataFetcher.fetch_front(self.instruments)


### Example Portfolio
class Trend(Portfolio):
    def __init__(self, instruments : list[Instrument], risk_target : float, capital : float):
        super().__init__()
        self.risk_object = GARCH(
            risk_target=risk_target,
            instruments=instruments,
            weights=(0.01, 0.01, 0.98),
            minimum_observations=100
        )
        self.weighted_strategies = [
            (1.0, TrendFollowing(instruments, self.risk_object, capital))
        ]
        self.capital = capital
        self.instruments = instruments
        self.portfolio_rules = [
            partial(
                dyn_opt,
                instrument_weights=equal_weight(instruments=instruments),
                cost_per_contract=3.0,
                asymmetric_risk_buffer=0.05,
                cost_penalty_scalar=10,
                position_limit_fn=position_limit(
                    max_leverage_ratio=2.0,
                    minimum_volume=100,
                    max_forecast_ratio=2.0,
                    max_forecast_buffer=0.5,
                    IDM=2.5,
                    tau=risk_target),
                portfolio_multiplier_fn=portfolio_multiplier(
                    max_portfolio_leverage=20,
                    max_correlation_risk=0.65,
                    max_portfolio_volatility=0.30,
                    max_portfolio_jump_risk=0.75)
            )
        ]

def calculate_correlation_risks_95th_percentile(portfolio):
    positions_df = portfolio.positions
    correlation_risks = []

    # Calculate weighted positions: (exposure * positions) / capital
    weighted_positions_df = (portfolio.exposure * positions_df) / portfolio.capital

    # Annualized volatilities from the risk object
    annualized_volatilities = portfolio.risk_object.get_var().to_standard_deviation().annualize().to_frame()

    # Ensure that weighted_positions_df and annualized_volatilities have the same shape
    assert weighted_positions_df.shape == annualized_volatilities.shape, "Mismatch between weighted positions and volatilities dataframes."

    # Iterate through the rows of the weighted positions dataframe
    for index, weighted_position_row in weighted_positions_df.iterrows():
        # Multiply the absolute value of the weighted position row by the corresponding row in annualized_volatilities
        weighted_risk = np.sum(np.abs(weighted_position_row) * annualized_volatilities.iloc[index])

        # Append the calculated weighted risk to correlation_risks
        correlation_risks.append(weighted_risk)

    # Calculate the 95th percentile for correlation risks after the loop
    correlation_risks_percentile_95 = np.percentile(correlation_risks, 95)

    return correlation_risks_percentile_95




def main():
    futures : list[Future] = [
        Future(symbol="ES", dataset="CME", multiplier=5)
    ]

    trend: Trend = Trend(futures, 0.2, 100_000)
    calculate_95th_percentile_per_multiplier(trend)

if __name__ == "__main__":
    main()