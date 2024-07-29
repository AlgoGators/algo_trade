# Function to start Xvfb
start_xvfb() {
    echo "> Starting Xvfb..."
    DISPLAY=:0
    export DISPLAY
    rm -f /tmp/.X1-lock
    /usr/bin/Xvfb $DISPLAY -ac -screen 0 1024x768x24 &
}

# Function to start socat
start_socat() {
    # run socat
	if [ -z "${SOCAT_PORT}" ]; then
		echo "> SOCAT_PORT not set, port: ${SOCAT_PORT}"
		exit 1
	fi
	if [ -n "$(pgrep -f "fork TCP:127.0.0.1:${API_PORT}")" ]; then
		# if this script is already running don't start it
		echo "> socat already active. Not starting a new one"
		return 0
	else
        echo "> Starting socat..."
        while true; do
            printf "Forking :::%d onto 0.0.0.0:%d > trading mode %s \n" \
                "${API_PORT}" "${SOCAT_PORT}" "${TRADING_MODE}"
            socat TCP-LISTEN:"${SOCAT_PORT}",fork TCP:127.0.0.1:"${API_PORT}"
            sleep 5
        done
    fi
}

# Function to start IB Controller
start_IBC() {
    echo "> Starting IB Controller..."
    if ! "./ibc/gatewaystart.sh"; then
        echo "Error: Failed to start IB Controller"
        exit 1
    fi
    pid=$!
    export pid
    echo "$pid" > "/tmp/pid_${TRADING_MODE}"
}

stop_ibc() {
	echo "> Received SIGINT or SIGTERM. Shutting down IB Gateway."
	# Stop Xvfb
	echo "> Stopping Xvfb."
	pkill Xvfb
    # Stop socat
    echo "> Stopping socat."
    pkill run_socat.sh
    pkill socat
	# Set TERM
	echo "> Stopping IBC."
	kill -SIGTERM "${pid[@]}"
	# Wait for exit
	wait "${pid[@]}"
	# All done.
	echo "> Done... $?"
}

fill_template() {
    # Read environment variables from .env file
    if [ -f /root/ibc/.env ]; then
        export $(cat /root/ibc/.env | xargs)
    fi

    # Define output file path
    output_file="/root/ibc/config.ini"

    # Check if the template file exists
    if [ ! -f "/root/ibc/config.ini.template" ]; then
        echo "> Template file '/root/ibc/config.ini.template' not found."
        exit 1
    fi

    # Perform variable substitution
    sed -e "s|\${IBKR_USERNAME}|${IBKR_USERNAME}|g" \
        -e "s|\${IBKR_PASSWORD}|${IBKR_PASSWORD}|g" \
        -e "s|\${TRADING_MODE}|${TRADING_MODE}|g" \
        -e "s|\${READ_ONLY_API}|${READ_ONLY_API}|g" \
        "/root/ibc/config.ini.template" > "$output_file"

    echo "> Configuration file '$output_file' generated successfully."
}

main() {
    fill_template

    # Start Xvfb
    start_xvfb

    # Start socat
    start_socat &

    # Start IB Controller
    start_IBC
}

main

trap stop_ibc SIGINT SIGTERM
wait "${pid[@]}"
exit $?