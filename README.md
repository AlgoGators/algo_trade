# Algo-Trade
![workflow](https://github.com/AlgoGators/algo_trade/actions/workflows/tests.yml/badge.svg)

## Poetry
### Installing Poetry
The easiest way to install Poetry is through Pipx, which is likely not already installed.

#### Windows:
```cmd
// Installing pip x
>> py -m pip install --user pipx

// It will likely raise the following warning:
WARNING: The script pipx.exe is installed in `<USER folder>\AppData\Roaming\Python\Python3x\Scripts` which is not on PATH

// Run the following:
>> cd userFolder\AppData\Roaming\Python\Python3x\Scripts
>> .\pipx.exe ensurepath

// Refresh terminal

// Installing poetry
>> pipx install poetry
```

or 

```cmd
>> python -m pip install --user pipx
>> python -m pipx ensurepath
```

#### Linux
```cmd
>> python3 -m pip install --user pipx
>> python3 -m pipx ensurepath
```

#### macOS
```cmd
>> python3 -m pip install --user pipx
>> python3 -m pipx ensurepath
```

---
### Using Poetry
```cmd
// Installing dependencies
>> poetry install

// Running main.py (from root directory)
>> poetry run python algo_trade/main.py

// OR (if the file has been added to pyproject.toml)
>> poetry run algo-trade
```

## Configuration:

Create a .env file with the following line of code, replacing `...` with the proper key:
```.env
DATABENTO_API_KEY=...
```
