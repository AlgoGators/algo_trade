# Algo-Trade

Algo Trade is a Python application... 

## Running ib-gateway:
This relies on [gnzsnz](https://github.com/gnzsnz)'s [ib-gateway-docker](https://github.com/gnzsnz/ib-gateway-docker) repository.
1. Clone the repository to desired machine.
2. Place contents of the repository into algo_trade/ibc
3. Create a .env in the root of the repository.
4. Inside this file, copy the following, change userName and passWord:
```
TWS_USERID=userName
TWS_PASSWORD=passWord
# ib-gateway
#TWS_SETTINGS_PATH=/home/ibgateway/Jts
# tws
#TWS_SETTINGS_PATH=/config/tws_settings
TWS_SETTINGS_PATH=
TWS_ACCEPT_INCOMING=
TRADING_MODE=paper
READ_ONLY_API=no
VNC_SERVER_PASSWORD=myVncPassword
TWOFA_TIMEOUT_ACTION=restart
BYPASS_WARNING=
AUTO_RESTART_TIME=11:59 PM
AUTO_LOGOFF_TIME=
TWS_COLD_RESTART=
SAVE_TWS_SETTINGS=
RELOGIN_AFTER_2FA_TIMEOUT=yes
TIME_ZONE=Europe/Zurich
CUSTOM_CONFIG=
SSH_TUNNEL=
SSH_OPTIONS=
SSH_ALIVE_INTERVAL=
SSH_ALIVE_COUNT=
SSH_PASSPHRASE=
SSH_REMOTE_PORT=
SSH_USER_TUNNEL=
SSH_RESTART=
SSH_VNC_PORT=
```
4. Run docker-compose up

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