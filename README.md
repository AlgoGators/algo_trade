# AlgoTrade - Advanced Algorithmic Trading Framework
![workflow](https://github.com/AlgoGators/algo_trade/actions/workflows/tests.yml/badge.svg)

## Overview
AlgoTrade is a sophisticated Python-based algorithmic trading framework designed for developing, testing, and deploying systematic trading strategies with a focus on futures markets. The framework provides robust risk management, dynamic optimization, and seamless integration with Interactive Brokers (IB) Gateway.

## Key Features
- **Advanced Risk Management**
  - GARCH volatility modeling
  - Correlation-based risk scaling
  - Position limits and portfolio-level risk controls
  - Jump risk detection and management

- **Dynamic Strategy Optimization**
  - Cost-aware position optimization
  - Asymmetric risk buffering
  - Transaction cost analysis
  - Portfolio rebalancing with tracking error minimization

- **Market Data Integration**
  - Databento integration for historical data
  - Real-time data through IB Gateway
  - Multi-asset class support (currently focused on futures)
  - Automated contract rolling

- **Trading System Architecture**
  - Modular strategy development framework
  - Event-driven execution engine
  - Asynchronous data handling
  - Comprehensive position and PnL tracking

## Installation

### Prerequisites
- Python 3.11+
- Poetry for dependency management
- Interactive Brokers Gateway (for live trading)

### Installing Poetry

#### Windows
```cmd
// Install pipx
py -m pip install --user pipx

// Add to PATH (if warning appears)
cd userFolder\AppData\Roaming\Python\Python3x\Scripts
.\pipx.exe ensurepath

// Install poetry
pipx install poetry
```

#### Linux
```bash
>> python3 -m pip install --user pipx
>> python3 -m pipx ensurepath
```

#### macOS
```zsh
>> python3 -m pip install --user pipx
>> python3 -m pipx ensurepath
```

---
### Using Poetry
```bash
poetry install

// Running main.py (from root directory)
poetry run python algo_trade/main.py

// OR (if the file has been added to pyproject.toml)
poetry run algo-trade
```
