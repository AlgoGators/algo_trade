import asyncio
from algo_trade.implementations import Trend
from algo_trade.instrument import Future

async def main():
    futures: list[Future] = [
        Future(symbol="ES", dataset="CME", multiplier=5)
    ]

    trend: Trend = Trend(futures, 0.2, 100_000)
    await trend.fetch_data()  # This will now be an asynchronous call

    print(trend.positions)
    print(trend.PnL.get(PnL.ReturnType.PERCENT, PnL.Timespan.CUMULATIVE))

if __name__ == "__main__":
    asyncio.run(main())