from ibapi import contract

class Contract(contract.Contract):
    def __init__(
            self,
            conId : int = None,
            symbol : str = None,
            secType : str = None,
            lastTradeDateOrContractMonth : str = None,
            multiplier : str = None,
            exchange : str = None,
            currency : str = None,
            contract : contract.Contract = None) -> None:
        super().__init__()
        self.conId : int = conId if conId is not None else self.conId
        self.symbol : str = symbol if symbol is not None else self.symbol
        self.secType : str = secType if secType is not None else self.secType
        self.lastTradeDateOrContractMonth : str = lastTradeDateOrContractMonth if lastTradeDateOrContractMonth is not None else self.lastTradeDateOrContractMonth
        self.multiplier : str = multiplier if multiplier is not None else self.multiplier
        self.exchange : str = exchange if exchange is not None else self.exchange
        self.currency : str = currency if currency is not None else self.currency

        if contract is not None:
            self.conId = contract.conId
            self.symbol = contract.symbol
            self.secType = contract.secType
            self.lastTradeDateOrContractMonth = contract.lastTradeDateOrContractMonth
            self.multiplier = contract.multiplier
            self.exchange = contract.exchange
            self.currency = contract.currency

    def __str__(self):
        return f"{self.conId},{self.symbol},{self.exchange},{self.multiplier},{self.currency},{self.secType}"
    
    def __hash__(self) -> int:
        return hash((self.symbol, self.exchange, self.multiplier, self.currency, self.conId, self.secType))

    def __repr__(self):
        return f"'{self.conId},{self.symbol},{self.exchange},{self.multiplier},{self.currency},{self.secType}'"
    
    def __eq__(self, other : 'Contract'):
        return self.symbol == other.symbol and self.exchange == other.exchange and self.multiplier == other.multiplier and self.currency == other.currency and self.conId == other.conId and self.secType == other.secType