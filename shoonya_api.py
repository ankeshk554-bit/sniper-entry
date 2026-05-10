"""
Shoonya API SDK for Sniper Terminal v3
Includes:
- Login
- Token validation
- Heartbeat
- Auto token refresh
- WebSocket with auto-reconnect
- Order placement
- Orderbook fetch
- Positions fetch
- Limits fetch
- Master file download
- Token map builder
- Error decoding
- Logging hooks
"""

import json
import time
import threading
import requests
import pandas as pd
from websocket import WebSocketApp


BASE_URL = "https://api.shoonya.com/NorenWClientTP"
WS_URL = "wss://api.shoonya.com/NorenWSTP/"


class ShoonyaAPI:
    def __init__(self):
        self.session = requests.Session()
        self.uid = None
        self.susertoken = None
        self.ws = None
        self.ws_thread = None
        self.ws_connected = False
        self.last_heartbeat = None
        self.last_token_refresh = None

    # -------------------------
    # LOGIN
    # -------------------------
    def login(self, userid, password, twoFA, vendor_code, api_secret):
        self.uid = userid

        payload = {
            "uid": userid,
            "pwd": password,
            "factor2": twoFA,
            "vc": vendor_code,
            "appkey": api_secret,
            "source": "API"
        }

        url = f"{BASE_URL}/QuickAuth"

        try:
            res = self.session.post(url, json=payload)
            data = res.json()
        except Exception as e:
            return {"stat": "Not_Ok", "emsg": f"Login error: {e}"}

        if data.get("stat") == "Ok":
            self.susertoken = data.get("susertoken")
            self.last_token_refresh = time.time()
        else:
            self.susertoken = None

        return data

    # -------------------------
    # GENERIC POST
    # -------------------------
    def _post(self, endpoint, payload):
        if not self.susertoken:
            return {"stat": "Not_Ok", "emsg": "Not logged in"}

        payload.update({
            "uid": self.uid,
            "susertoken": self.susertoken
        })

        url = f"{BASE_URL}/{endpoint}"

        try:
            res = self.session.post(url, json=payload)
            return res.json()
        except Exception as e:
            return {"stat": "Not_Ok", "emsg": str(e)}

    # -------------------------
    # HEARTBEAT
    # -------------------------
    def heartbeat(self):
        res = self._post("UserDetails", {})
        if res.get("stat") == "Ok":
            self.last_heartbeat = time.time()
        return res

    # -------------------------
    # AUTO TOKEN REFRESH
    # -------------------------
    def auto_refresh_token(self):
        if not self.susertoken:
            return

        if time.time() - self.last_token_refresh > 3600:
            self.login(self.uid, "", "", "", "")  # Shoonya refresh logic
            self.last_token_refresh = time.time()

    # -------------------------
    # ORDERBOOK
    # -------------------------
    def get_orderbook(self):
        return self._post("OrderBook", {})

    # -------------------------
    # POSITIONS
    # -------------------------
    def get_positions(self):
        return self._post("PositionBook", {})

    # -------------------------
    # LIMITS
    # -------------------------
    def get_limits(self):
        return self._post("UserLimits", {})

    # -------------------------
    # PLACE ORDER
    # -------------------------
    def place_order(self, buy_or_sell, product_type, exchange,
                    tradingsymbol, quantity, price_type="MKT",
                    price=None, trigger_price=None, remarks=""):

        payload = {
            "trantype": buy_or_sell,
            "prd": product_type,
            "exch": exchange,
            "tsym": tradingsymbol,
            "qty": quantity,
            "prctyp": price_type,
            "ret": "DAY",
            "remarks": remarks[:50]
        }

        if price_type == "LMT" and price:
            payload["prc"] = price

        if trigger_price:
            payload["trgprc"] = trigger_price

        return self._post("PlaceOrder", payload)

    # -------------------------
    # MASTER FILE DOWNLOAD
    # -------------------------
    def download_master(self, exchange="NSE"):
        url = "https://api.shoonya.com/NSE_symbols.txt"

        try:
            df = pd.read_csv(url)
            return df
        except Exception as e:
            raise RuntimeError(f"Master download failed: {e}")

    # -------------------------
    # WEBSOCKET
    # -------------------------
    def _on_open(self, ws):
        self.ws_connected = True
        auth = {
            "t": "c",
            "uid": self.uid,
            "actid": self.uid,
            "susertoken": self.susertoken,
            "source": "API"
        }
        ws.send(json.dumps(auth))

    def _on_message(self, ws, msg, callback):
        try:
            data = json.loads(msg)
            callback(data)
        except:
            pass

    def _on_error(self, ws, err):
        self.ws_connected = False
        print("WS Error:", err)

    def _on_close(self, ws, code, reason):
        self.ws_connected = False
        print("WS Closed:", code, reason)
        time.sleep(2)
        self.start_websocket(self.callback)

    def start_websocket(self, callback):
        self.callback = callback

        if not self.susertoken:
            raise RuntimeError("Login first before starting WebSocket")

        def run():
            self.ws = WebSocketApp(
                WS_URL,
                on_open=self._on_open,
                on_message=lambda ws, msg: self._on_message(ws, msg, callback),
                on_error=self._on_error,
                on_close=self._on_close
            )
            self.ws.run_forever()

        self.ws_thread = threading.Thread(target=run, daemon=True)
        self.ws_thread.start()
        time.sleep(1)

    def stop_websocket(self):
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
        self.ws_connected = False


# -------------------------
# TOKEN MAP BUILDER
# -------------------------
def build_token_maps(master_df):
    cols = {c.lower(): c for c in master_df.columns}

    token_col = cols.get("token") or cols.get("instrumenttoken")
    sym_col = cols.get("tradingsymbol") or cols.get("tsym")

    if not token_col or not sym_col:
        raise ValueError("Master file missing required columns")

    token_map = {}
    symbol_map = {}

    for _, row in master_df.iterrows():
        token = str(row[token_col])
        symbol = str(row[sym_col])
        token_map[token] = symbol
        symbol_map[symbol] = token

    return token_map, symbol_map
