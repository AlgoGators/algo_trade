import logging
import ipaddress

from algo_trade.ib_utils._config import LOCALHOST, PORT, CLIENT_ID
from algo_trade.ib_utils._contract import Contract
from algo_trade.ib_utils.api_handler import api_handler_context
from algo_trade.ib_utils.error_codes import NoSecurityFound
from algo_trade.instrument import Instrument

def validate_contracts(instruments : list[Instrument]) -> None:
    with api_handler_context(ipaddress.ip_address(LOCALHOST), PORT, CLIENT_ID) as api_handler: 
        not_found = []
        for instrument in instruments:
            contract = Contract.from_instrument(instrument)
            try:
                api_handler.get_contract_details(contract)
            except NoSecurityFound as e:
                not_found.append(contract)
        
        for contract in not_found:
            logging.error(f"Contract {contract} not found")
    
        if not_found:
            raise ValueError(f"Contracts not found: {not_found}")
