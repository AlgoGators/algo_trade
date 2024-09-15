# Algo-Trade

## IBAPI Downtimes (from [IBKR](https://www.interactivebrokers.com/en/?f=%2Fen%2Fsoftware%2FsystemStatus.php)):
| Server Reset Times  | North America                | Europe | Asia |
|---------------------|:----------------------------:|:------:|:----:|
| Saturday - Thursday | 23:45 - 00:45 ET<sup>1</sup> |  ...   |  ... |
| Friday              | 23:00 - 03:00 ET<sup>2</sup> |  ...   |  ... |


> Disclosures: 
> 1. The reset period describes the duration during which your account may be unavailable for a few seconds. It does not indicate that the entire system will be unavailable for the full reset period. During a reset period, there may be an interruption in the ability to log in or manage orders. Existing orders (native types) will operate normally although execution reports and simulated orders will be delayed until the reset is complete.
> 2. During the Friday evening reset period, all services will be unavailable in all regions for the duration of the reset.

---

![workflow](https://github.com/AlgoGators/algo_trade/actions/workflows/tests.yml/badge.svg)

## Poetry
### Installing Poetry
The easiest way to install Poetry is through Pipx, which is likely not already installed.

#### Windows:
```
// Installing pip x
>> py -m pip install --user pipx

// It will likely raise the following warning:
WARNING: The script pipx.exe is installed in `<USER folder>\AppData\Roaming\Python\Python3x\Scripts` which is not on PATH

// Run the following:
>> cd <USER folder>\AppData\Roaming\Python\Python3x\Scripts
>> .\pipx.exe ensurepath

// Refresh terminal

// Installing poetry
>> pipx install poetry
```

#### Linux
```
// Ask Cole
```

#### macOS
```
```

---
### Using Poetry
```
// Installing dependencies
>> poetry install

// Running main.py (from root directory)
>> poetry run python algo_trade/main.py

// OR (if the file has been added to pyproject.toml)
>> poetry run algo-trade
```

## API Config File:
Ensure config/config.toml exists with the following values:
```
[server]
ip = ""
user = ""
password = ""

[database]
demo = ""
db_trend = ""
db_carry = ""
db_test = ""
user = ""
password = ""
port = ""

[databento]
api_historical = ""
api_live = ""
```
