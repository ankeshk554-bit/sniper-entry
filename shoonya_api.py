class ShoonyaAPI:
    def __init__(self):
        pass

    def login(self, userid, password, twoFA, vendor_code, api_secret):
        # your login logic
        pass

    def start_websocket(self, callback):
        # your websocket logic
        pass

    def download_master(self, exchange):
        # your master file logic
        pass

    def place_order(self, **kwargs):
        # your order placement logic
        pass
def build_token_maps(master_df):
    token_map = {}
    symbol_map = {}

    for _, row in master_df.iterrows():
        token = str(row["Token"])
        symbol = row["TradingSymbol"]

        token_map[token] = symbol
        symbol_map[symbol] = token

    return token_map, symbol_map
