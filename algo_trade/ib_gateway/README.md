# IB Gateway Docker
## Step-by-step Instructions

### Installing IB Gateway (optional)
- Download the stable version for Linux x64 [here](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php)
- Place the file `ibgateway-stable-standalone-linux-x64.sh` inside the installer folder, naming it `ibgateway-<VERSION_NUMBER>-linux-x64.sh`

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

### Laptop Performance Issues
If you're on a laptop, or older desktop, and you noticed Docker is taking significant resources, try going to %UserProfile%\ and creating (if it does not exist) .wslconfig and adding to it:
```
[wsl2]
memory=2GB
```
Changing it as necessary to limit the total memory Docker will use.

### Accessing the IB-Gateway
Make sure you connect with the API on `127.0.0.1:4002` for paper trading (`:4001` for live trading—see __NOTE__ above).

# IB Gateway Utils
### NOTICE:
- ib_insync will __NOT__ be used.
- ibapi must be downloaded from the IBKR official github.io page or from this repository.

## IB API

### Installation
#### Windows:
```
cmd.exe
> cd PATH/TO/PROJECT/Lib
> python setup.py install
```

#### Linux (ask Cole R)
- Install [Intel Decimal Library](https://www.intel.com/content/www/us/en/developer/articles/tool/intel-decimal-floating-point-math-library.html)

#### MacOS
- GLHF

---

### Updating (requires separate PR)
1. Navigate to the [IBKR GitHub page](https://interactivebrokers.github.io/)
2. Follow the onscreen instructions and download the STABLE version for your OS
3. Run the installer, following onscreen instructions
4. Move the source/pythonclient/ibapi folder to this project in Lib/ibapi
5. Issue new PR with the new ibapi
 
### Documentation
The original GitHub.io page for the documentation is no longer in use; please use the new [website](https://ibkrcampus.com/ibkr-api-page/twsapi-doc/#notes-and-limitations).

## IB Gateway

### Headful Client
1. Download and run the installer here IBKR's [website](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php).
2. Launch IB Gateway
3. Log in, selecting IB API & Paper Trading
4. Go to Configure -> Settings -> API 
5. Uncheck Read-Only API

### Headless Client
See above
