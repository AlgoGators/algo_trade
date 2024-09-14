from enum import Enum, StrEnum
from algo_trade.instrument import Future

class SecurityType(Enum):
    FUTURE = (Future, 'FUT')

    def __init__(self, obj, string):
        self.obj = obj
        self.string = string

    @classmethod
    def from_str(cls, value: str) -> "SecurityType":
        """
        Converts a string to a SecurityType enum based on the value to the Enum name and not value
        so "FUTURE" -> FUTURE

        Args:
            - value: str - The value to convert to a SecurityType enum

        Returns:
            - SecurityType: The SecurityType enum
        """
        try:
            return cls[value.upper()]
        except ValueError:
            # If exact match fails, look for a case-insensitive match
            for member in cls:
                if member.name.lower() == value.lower():
                    return member

            raise ValueError(f"{value} is not a valid {cls.__name__}")
        
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