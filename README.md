# Algo-Trade
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

or 

```
>> python -m pip install --user pipx
>> python -m pipx ensurepath
```

#### Linux
```
>> python3 -m pip install --user pipx
>> python3 -m pipx ensurepath
```

#### macOS
```
>> python3 -m pip install --user pipx
>> python3 -m pipx ensurepath
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
