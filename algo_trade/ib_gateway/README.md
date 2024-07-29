# IB Gateway Docker
## Step-by-step Instructions

### Clone this repository
`gh repo clone AlgoGators/ib-gateway`

### Installing IBC
- Download the 3.18.0 version for Linux [here](https://github.com/IbcAlpha/IBC/releases/tag/3.18.0-Update.1)
- Unzip the folder and place it in this repository (naming it `ibc`)

### Installing IB Gateway (optional)
- Download the stable version for Linux x64 [here](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php)
- Place the file `ibgateway-stable-standalone-linux-x64.sh` inside the installer folder, naming it `ibgateway-<VERSION_NUMBER>-linux-x64.sh`

### Creating .env file with Private Settings
Create a file in the directory named .env, it should look like this:
```
IBKR_USERNAME=USERNAME
IBKR_PASSWORD=PASSWORD
API_PORT=4002
SOCAT_PORT=4004
TRADING_MODE=paper
READ_ONLY_API=no
```

Make sure to update the line in the dockerfile for the current version of IB Gateway: 
`ENV IB_GATEWAY_VERSION=1019`

#### NOTE:
Only paper trading is support right now, so make sure:
```
API_PORT=4002 # Port IB Gateway listens to for paper trading
SOCAT_PORT=4004
TRADING_MODE=paper
```
If you wish to use live trading, you must make the settings:
```
API_PORT=4001 # Port IB Gateway listens to for live trading
SOCAT_PORT=4003
TRADING_MODE=live
```
And include in your docker-compose file:
```
ports:
    - "127.0.0.1:4001:4003" # Expose IB Gateway live trading port
```
__BOTH__ paper and live trading is not supported in any form as of yet.

### Building the Docker container
From the repository directory:

To build:
```
docker build -f Dockerfile.ib-gateway -t ib-gateway .
```

To run:
```
docker run --env .env -d -p 127.0.0.1:4002:4004 ib-gateway
```

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
