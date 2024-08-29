from functools import partial

from algo_trade.portfolio import Portfolio
from algo_trade.strategy import Strategy
from algo_trade.instrument import Instrument, Future, RollType, ContractType, Agg
from algo_trade.rules import capital_scaling, risk_parity, equal_weight, trend_signals
from algo_trade.risk_measures import GARCH

class TrendFollowing(Strategy[Future]):
    def __init__(self, instruments: list[Future], risk_target: float, capital: float):
        super().__init__(capital=capital)
        # Overload the instruments
        self.instruments: list[Future] = instruments
        self.risk_object = GARCH(
            risk_target=risk_target,
            instruments=instruments,
            weights=(0.01, 0.01, 0.98),
            minimum_observations=100
        )
        
        self.rules = [
            partial(risk_parity, risk_object=self.risk_object),
            partial(trend_signals, instruments=instruments, risk_object=self.risk_object),
            partial(equal_weight, instruments=instruments),
            partial(capital_scaling, instruments=instruments, capital=capital)
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
        [instrument.add_data(Agg.DAILY, RollType.CALENDAR, ContractType.FRONT) for instrument in self.instruments]


### Example Portfolio
class Trend(Portfolio):
    def __init__(self, instruments : list[Instrument], risk_target : float, capital : float):
        self.strategies = [
            (1.0, TrendFollowing(instruments, risk_target, capital))
        ]
        super().__init__(instruments, self.strategies, capital)

def main():
    import pandas as pd
    instruments: list[Future] = [
        Future(symbol="ES", dataset="CME", multiplier=5)
    ]
    trend_following: TrendFollowing = TrendFollowing(instruments, 0.2, 100_000)
    positions: pd.DataFrame = trend_following.positions
    print(positions)


if __name__ == "__main__":
    main()
