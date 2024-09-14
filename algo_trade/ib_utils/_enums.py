from enum import StrEnum
        
class AccountSummaryTag(StrEnum):
    ACCOUNT_TYPE = "AccountType"
    NET_LIQUIDATION = "NetLiquidation"
    TOTAL_CASH_VALUE = "TotalCashValue"
    SETTLED_CASH = "SettledCash"
    ACCRUED_CASH = "AccruedCash"
    BUYING_POWER = "BuyingPower"
    EQUITY_WITH_LOAN_VALUE = "EquityWithLoanValue"
    PREVIOUS_DAY_EQUITY_WITH_LOAN_VALUE = "PreviousDayEquityWithLoanValue"
    GROSS_POSITION_VALUE = "GrossPositionValue"
    REQ_T_EQUITY = "ReqTEquity"
    REQ_T_MARGIN = "ReqTMargin"
    SMA = "SMA"
    INIT_MARGIN_REQ = "InitMarginReq"
    MAINT_MARGIN_REQ = "MaintMarginReq"
    AVAILABLE_FUNDS = "AvailableFunds"
    EXCESS_LIQUIDITY = "ExcessLiquidity"
    CUSHION = "Cushion"
    FULL_INIT_MARGIN_REQ = "FullInitMarginReq"
    FULL_MAINT_MARGIN_REQ = "FullMaintMarginReq"
    FULL_AVAILABLE_FUNDS = "FullAvailableFunds"
    FULL_EXCESS_LIQUIDITY = "FullExcessLiquidity"
    LOOK_AHEAD_NEXT_CHANGE = "LookAheadNextChange"
    LOOK_AHEAD_INIT_MARGIN_REQ = "LookAheadInitMarginReq"
    LOOK_AHEAD_MAINT_MARGIN_REQ = "LookAheadMaintMarginReq"
    LOOK_AHEAD_AVAILABLE_FUNDS = "LookAheadAvailableFunds"
    LOOK_AHEAD_EXCESS_LIQUIDITY = "LookAheadExcessLiquidity"
    HIGHEST_SEVERITY = "HighestSeverity"
    DAY_TRADES_REMAINING = "DayTradesRemaining"
    LEVERAGE = "Leverage"
    LEDGER = "$LEDGER:ALL"

class CurrencyName(StrEnum):
    USD = 'USD'
    EUR = 'EUR'
    GBP = 'GBP'
    AUD = 'AUD'

class OrderAction(StrEnum):
    BUY = 'BUY'
    SELL = 'SELL'

class OrderType(StrEnum):
    MARKET = 'MKT'
    LIMIT = 'LMT'
    STOP = 'STP'
    STOP_LIMIT = 'STP LMT'

class AlgoStrategy(StrEnum):
    ADAPTIVE = 'Adaptive'
    TWAP = 'Twap'
    VWAP = 'Vwap'

class AdaptiveOrderPriority(StrEnum):
    URGENT = 'Urgent'
    PATIENT = 'Patient'
    NORMAL = 'Normal'

class ExchangeName(StrEnum):
    NASDAQ = 'NASDAQ'
    EUREX = 'EUREX'
    GLOBEX = 'GLOBEX'
    SMART = 'SMART'
    CME = 'CME'