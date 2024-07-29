# Algo-Trade

Algo Trade is a Python application... 

## Running ib-gateway:
This relies on [gnzsnz](https://github.com/gnzsnz)'s [ib-gateway-docker](https://github.com/gnzsnz/ib-gateway-docker) repository.
1. Clone the repository to desired machine.
2. Create a .env in the root of the reopository.
3. Inside this file, copy the following, change userName and passWord:
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