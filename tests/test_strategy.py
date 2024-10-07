import unittest

from algo_trade.strategy import Strategy

class SampleStrategy(Strategy):
    def __init__(self, capital):
        super().__init__()
        self.capital = capital

    async def fetch_data(self):
        pass

class TestStrategy(unittest.TestCase):
    def test_strategy(self):
        capital = 100000
        strategy = SampleStrategy(capital)
        self.assertEqual(strategy.capital, capital)
        self.assertEqual(strategy.instruments, None)
        self.assertEqual(strategy.risk_object, None)
        self.assertEqual(strategy.rules, None)
        self.assertEqual(strategy.scalars, (1.0,))

if __name__ == '__main__':
    unittest.main()