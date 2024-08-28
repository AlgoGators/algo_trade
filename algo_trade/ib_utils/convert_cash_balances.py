from src.api_handler import APIHandler
from src._config import LOCALHOST, PORT, CLIENT_ID

def convert_cash_balances(target_currency : str = 'USD') -> None:
    #@ UNCOMMENT api_handler = APIHandler(
    #@ UNCOMMENT     IP_ADDRESS = LOCALHOST,
    #@ UNCOMMENT     PORT=PORT,
    #@ UNCOMMENT     CLIENT_ID=CLIENT_ID
    #@ UNCOMMENT )

    #@ UNCOMMENT api_handler.connect()

    #@ UNCOMMENT api_handler.cancel_outstanding_orders()

    #@ UNCOMMENT cash_balances = api_handler.get_cash_balances()

    from decimal import Decimal #! REMOVE
    cash_balances = {'USD': Decimal('1000.00'), 'EUR': Decimal('2000.00')} #! REMOVE

    for currency, amount in cash_balances.items():
        if currency == target_currency:
            continue
        print(amount)

if __name__ == "__main__":
    convert_cash_balances()