import eel
import random  # For simulating market data

eel.init('web')  # Point to the 'web' directory

# Function to fetch market data (You would fetch real data using an API like Alpha Vantage, Yahoo Finance, etc.)
@eel.expose
def get_market_data():
    data = [
        {'symbol': 'ES', 'price': round(random.uniform(4000, 4500), 2), 'change': round(random.uniform(-10, 10), 2), 'volume': random.randint(1000, 5000)},
        {'symbol': 'NQ', 'price': round(random.uniform(15000, 16000), 2), 'change': round(random.uniform(-100, 100), 2), 'volume': random.randint(500, 3000)},
    ]
    return data

# Start the Eel application
eel.start('index.html', size=(1000, 600))
