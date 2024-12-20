from enum import IntEnum

class ErrorCodes(IntEnum):
    CANT_CONNECT_TO_TWS = 502
    OUTDATED_TWS = 503 #
    NOT_CONNECTED = 504 # Not really possible
    CONNECTIVITY_LOST = 1100 #
    MARKET_DATA_FARM_CONNECTION_LOST = 2103 #
    MARKET_DATA_FARM_CONNECTED = 2104
    HMDS_DATA_FARM_CONNECTION_LOST = 2105 #
    HMDS_DATA_FARM_CONNECTED = 2106
    HMDS_DATA_FARM_CONNECTED_INACTIVE = 2107
    MARKET_DATA_FARM_CONNECTED_INACTIVE = 2108
    CROSS_SIDE_WARNING = 2137 #
    SEC_DEF_DATA_FARM_CONNECTED = 2158
    NO_SECURITY_FOUND = 200

class NoSecurityFound(Exception):
    pass