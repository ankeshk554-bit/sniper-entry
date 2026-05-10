"""
shoonya_api.py
Lightweight Shoonya (Noren) API wrapper for Sniper Terminal v3
"""

import time
import json
import threading
import traceback
from typing import Callable, Tuple

import requests
import pandas as pd
from websocket import WebSocketApp


# =========================
# CONFIG
# =========================

SHOONYA_BASE_URL = "https://api.shoonya.com/NorenWClientTP"
USER_AGENT = "SniperTerminalV3"


# =========================
# CORE API WRAPPER
# =========================

class ShoonyaAPI:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

        self.uid = None
        self.susertoken = None
        self.ws: WebSocketApp | None = None
        self.ws_thread: threading.Thread | None = None
        self.ws_connected = False

    # -------------
    # LOGIN
    # -------------
    def login(self, userid: str, password: str, twoFA: str,
              vendor_code: str, api_secret: str) -> dict:
        """
        Perform Shoonya QuickAuth login.
        Returns the raw JSON response.
        """

        self.uid = userid

        payload = {
            "uid": userid,
            "pwd": password,
            "factor2": twoFA,
            "vc": vendor_code,
            "appkey": api_secret,
            "source": "API",
        }

        url = f"{SHOONYA_BASE_URL}/QuickAuth"
        res = self.session.post(url, json=payload)
        try:
            data = res.json()
        except Exception:
            data = {"stat": "Not_Ok", "emsg": f"Invalid JSON from Shoonya: {res.text}"}
            return data

        if data.get("stat") == "Ok":
            self.susertoken = data.get("susertoken")
        else:
            self.susertoken = None

        return data

    # -------------
    # GENERIC REQUEST
    # -------------
    def _post(self, endpoint: str, payload: dict) -> dict:
        if not self.susertoken:
            return {"stat": "Not_Ok", "emsg": "Not logged in / susertoken missing"}

        url = f"{SHOONYA_BASE_URL}/{endpoint}"
        payload = payload.copy()
        payload.update({
            "uid": self.uid,
            "susertoken": self.susertoken,
        })

        try:
            res = self.session.post(url, json=payload, timeout=10)
            return res.json()
        except Exception as e:
            return {"stat": "Not_Ok", "emsg": f"Request error: {e}"}

    # -------------
    # LIMITS (TEST CALL)
    # -------------
    def get_limits(self) -> dict:
        """
        Simple test call to verify connection & login.
        """
        return self._post("UserLimits", {})

    # -------------
    # MASTER DOWNLOAD
    # -------------
    def download_master(self, exchange: str = "NSE") -> pd.DataFrame:
        """
        Download Shoonya master file for an exchange.
        Many brokers provide a URL for this; if your broker
        gives a direct CSV link, plug it here.
        For now, this is a placeholder that expects a CSV URL.
        """

        # TODO: Replace this with your actual master URL
        # Example (NOT REAL):
        # master_url = f"https://api.shoonya.com/{exchange}_symbols.csv"

        raise NotImplementedError(
            "download_master() must be implemented with your broker's master file URL."
        )

    # -------------
    # PLACE ORDER
    # -------------
    def place_order(
        self,
        buy_or_sell: str,
        product_type: str,
        exchange: str,
        tradingsymbol: str,
        quantity: int,
        price_type: str = "MKT",
        price: float | None = None,
        trigger_price: float | None = None,
        remarks: str = "",
    ) -> dict:
        """
        Place an order via Shoonya.
        buy_or_sell: 'B' or 'S'
        product_type: 'I', 'M', etc.
        price_type: 'MKT' or 'LMT'
        """

        payload = {
            "trantype": buy_or_sell,
            "prd": product_type,
            "exch": exchange,
            "tsym": tradingsymbol,
            "qty": quantity,
            "prctyp": price_type,
            "ret": "DAY",
            "remarks": remarks[:50],
        }

        if price_type == "LMT" and price is not None:
            payload["prc"] = price

        if trigger_price is not None:
            payload["trgprc"] = trigger_price

        return self._post("PlaceOrder", payload)

    # -------------
    # WEBSOCKET
    # -------------
    def _on_ws_open(self, ws: WebSocketApp):
        self.ws_connected = True

        # Send auth message
        auth_msg = {
            "t": "c",
            "uid": self.uid,
            "actid": self.uid,
            "susertoken": self.susertoken,
            "source": "API",
        }
        ws.send(json.dumps(auth_msg))

    def _on_ws_message(self, ws: WebSocketApp, message: str, callback: Callable):
        try:
            msg = json.loads(message)
        except Exception:
            return

        try:
            callback(msg)
        except Exception:
            traceback.print_exc()

    def _on_ws_error(self, ws: WebSocketApp, error):
        self.ws_connected = False
        print("Shoonya WS Error:", error)

    def _on_ws_close(self, ws: WebSocketApp, code, reason):
        self.ws_connected = False
        print("Shoonya WS Closed:", code, reason)

    def start_websocket(self, callback: Callable[[dict], None]):
        """
        Start Shoonya WebSocket and route ticks to `callback(msg: dict)`.
        """

        if not self.susertoken or not self.uid:
            raise RuntimeError("Cannot start WebSocket: not logged in.")

        if self.ws_connected:
            return  # already running

        # Shoonya WS URL (Noren)
        ws_url = "wss://api.shoonya.com/NorenWSTP/"

        def _run():
            self.ws = WebSocketApp(
                ws_url,
                on_open=self._on_ws_open,
                on_message=lambda ws, msg: self._on_ws_message(ws, msg, callback),
                on_error=self._on_ws_error,
                on_close=self._on_ws_close,
            )
            self.ws.run_forever()

        self.ws_thread = threading.Thread(target=_run, daemon=True)
        self.ws_thread.start()

        # Give it a moment to connect
        time.sleep(1)

    def stop_websocket(self):
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
        self.ws_connected = False


# =========================
# TOKEN MAP BUILDER
# =========================

def build_token_maps(master_df: pd.DataFrame) -> Tuple[dict, dict]:
    """
    Build token_map and symbol_map from Shoonya master DataFrame.

    Expected columns (adjust if your master file differs):
      - 'Token' or 'token'
      - 'TradingSymbol' or 'tsym'
    """

    # Try to be flexible with column names
    cols = {c.lower(): c for c in master_df.columns}

    token_col = cols.get("token") or cols.get("instrumenttoken") or cols.get("tokenid")
    sym_col = cols.get("tradingsymbol") or cols.get("tsym") or cols.get("symbol")

    if not token_col or not sym_col:
        raise ValueError(
            f"Master file missing required columns. Got: {list(master_df.columns)}"
        )

    token_map = {}
    symbol_map = {}

    for _, row in master_df.iterrows():
        token = str(row[token_col])
        symbol = str(row[sym_col])

        token_map[token] = symbol
        symbol_map[symbol] = token

    return token_map, symbol_map
