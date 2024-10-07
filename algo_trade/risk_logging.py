import csv
import datetime
import io
import logging
from enum import Enum
from typing import Optional


class StringEnum(str, Enum):
    def __repr__(self) -> str: return str(self.value)

class LogType(StringEnum):
    POSITION_LIMIT = "POSITION LIMIT"
    PORTFOLIO_MULTIPLIER = "PORTFOLIO MULTIPLIER"

class LogSubType(StringEnum):
    MAX_LEVERAGE = "MAX LEVERAGE"
    MAX_FORECAST = "MAX FORECAST"
    MAX_OPEN_INTEREST = "MAX OPEN INTEREST"
    LEVERAGE_MULTIPLIER = "LEVERAGE_MULTIPLIER"
    CORRELATION_MULTIPLIER = "CORRELATION_MULTIPLIER"
    VOLATILITY_MULTIPLIER = "VOLATILITY_MULTIPLIER"
    JUMP_MULTIPLIER = "JUMP_MULTIPLIER"
    MINIMUM_VOLUME = "MINIMUM VOLUME"

class LogMessage():
    _date : str | datetime.datetime
    _type : LogType
    _subtype : Optional[LogSubType]
    _info : Optional[str]
    _additional_info : Optional[str]
    def __init__(
            self,
            DATE : datetime.datetime,
            TYPE : LogType,
            SUBTYPE : Optional[LogSubType] = None,
            INFO : Optional[str] = None,
            ADDITIONAL_INFO : Optional[str] = None):
        self._date = DATE
        self._type = TYPE
        self._subtype = SUBTYPE
        self._info = INFO
        self._additional_info = ADDITIONAL_INFO
        self.message = [self._date, self._type, self._subtype, self._info, self._additional_info]

    @classmethod
    def attrs(cls) -> list[str]:
        keys = cls.__annotations__.keys()
        return [x.strip('_') for x in list(keys)]

    def __str__(self) -> str:
        return str(self.message)
    
    def __repr__(self) -> str:
        return str(self.message)

class CsvFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__()
        self.output = io.StringIO()
        self.writer = csv.writer(self.output, quoting=csv.QUOTE_ALL)
        self.write_header()
    
    def format(self, record : logging.LogRecord) -> str:
        if not isinstance(record.msg, LogMessage):
            return super().format(record)

        row = [record.levelname]
        row.extend(str(item) for item in record.msg.message)
        self.writer.writerow(row)
        data = self.output.getvalue()
        self.output.truncate(0)
        self.output.seek(0)
        return data.strip()

    def write_header(self) -> None:
        header = ['Level']
        header.extend(LogMessage.attrs())
        self.output.write(','.join(map(str, header)))
        self.output.write('\n')
