from decimal import Decimal
import typing
from ..src._contract import Contract

class ContractDetails:
    contract : Contract
    marketName : str
    minTick : float
    orderTypes : str
    validExchanges : str
    priceMagnifier : int
    underConId : int
    longName : str
    contractMonth : str
    industry : str
    category : str
    subcategory : str
    timeZoneId : str
    tradingHours : str
    liquidHours : str
    evRule : str
    evMultiplier : int
    aggGroup : int
    underSymbol : str
    underSecType : str
    marketRuleIds : str
    secIdList : typing.Optional[str]
    realExpirationDate : str
    lastTradeTime : str
    stockType : str
    minSize : Decimal
    sizeIncrement : Decimal
    suggestedSizeIncrement : Decimal
    cusip : str
    ratings : str
    descAppend : str
    bondType : str
    couponType : str
    callable : bool
    putable : bool
    coupon : int
    convertible : bool
    maturity : str
    issueDate : str
    nextOptionDate : str
    nextOptionType : str
    nextOptionPartial : bool
    notes : str