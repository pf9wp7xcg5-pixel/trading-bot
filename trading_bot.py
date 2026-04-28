"""
Bot de trading algorithmique — Stratégie RSI + Moyennes Mobiles
Exchange : Kraken (via ccxt)
Mode : Paper Trading activé par défaut (aucun ordre réel)

Installation :
    pip install ccxt pandas ta
"""

import time
import logging
import os
from datetime import datetime
import ccxt
import pandas as pd
import ta

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

API_KEY    = os.environ.get("KRAKEN_API_KEY", "")
API_SECRET = os.environ.get("KRAKEN_API_SECRET", "")

SYMBOL          = "BTC/USDC"
TIMEFRAME       = "1h"
CAPITAL_USDC    = 32.0
RISK_PER_TRADE  = 0.02
STOP_LOSS_PCT   = 0.03
TAKE_PROFIT_PCT = 0.06
MAX_DRAWDOWN    = 0.10
RSI_BUY         = 40
RSI_SELL        = 65

PAPER_TRADING   = os.environ.get("PAPER_TRADING", "true").lower() == "true"
LOOP_INTERVAL   = 60 * 60

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trading_bot.log"),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  ÉTAT INTERNE
# ─────────────────────────────────────────────

state = {
    "position"       : None,
    "entry_price"    : 0.0,
    "quantity"       : 0.0,
    "capital"        : CAPITAL_USDC,
    "capital_initial": CAPITAL_USDC,
    "trade_count"    : 0,
    "pnl_total"      : 0.0,
}

# ─────────────────────────────────────────────
#  CONNEXION KRAKEN
# ─────────────────────────────────────────────

def connect() -> ccxt.kraken:
    exchange = ccxt.kraken({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "timeout": 30000,
        "enableRateLimit": True,
    })
    log.info("Connexion Kraken établie.")
    return exchange

# ─────────────────────────────────────────────
#  DONNÉES DE MARCHÉ
# ─────────────────────────────────────────────

def get_ohlcv(exchange, limit: int = 250) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["close"] = df["close"].astype(float)
    df["high"]  = df["high"].astype(float)
    df["low"]   = df["low"].astype(float)
    return df

def get_price(exchange) -> float:
    ticker = exchange.fetch_ticker(SYMBOL)
    return float(ticker["last"])

# ─────────────────────────────────────────────
#  INDICATEURS
# ─────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["rsi"]   = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["ma50"]  = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()
    return df

# ─────────────────────────────────────────────
#  SIGNAUX
# ─────────────────────────────────────────────

def signal(df: pd.DataFrame) -> str:
    last = df.iloc[-1]
    prev = df.iloc[-2]

    bullish_trend    = last["ma50"] > last["ma200"]
    bearish_trend    = last["ma50"] < last["ma200"]
    rsi_oversold     = last["rsi"] < RSI_BUY
    rsi_overbought   = last["rsi"] > RSI_SELL
    rsi_crosses_up   = prev["rsi"] < RSI_BUY  and last["rsi"] >= RSI_BUY
    rsi_crosses_down = prev["rsi"] > RSI_SELL and last["rsi"] <= RSI_SELL

    if (rsi_oversold or rsi_crosses_up) and bullish_trend:
        return "BUY"
    if (rsi_overbought or rsi_crosses_down) and bearish_trend:
        return "SELL"
    return "HOLD"

# ─────────────────────────────────────────────
#  GESTION DU RISQUE
# ─────────────────────────────────────────────

def position_size(capital: float, price: float) -> float:
    risk_amount   = capital * RISK_PER_TRADE
    stop_distance = price * STOP_LOSS_PCT
    qty = risk_amount / stop_distance
    return round(qty, 6)

def check_drawdown() -> bool:
    loss_pct = (state["capital_initial"] - state["capital"]) / state["capital_initial"]
    if loss_pct >= MAX_DRAWDOWN:
        log.warning(f"Drawdown maximum atteint ({loss_pct:.1%}). Arrêt du bot.")
        return True
    return False

# ─────────────────────────────────────────────
#  ORDRES
# ─────────────────────────────────────────────

def buy(exchange, price: float) -> None:
    qty  = position_size(state["capital"], price)
    cost = qty * price

    if cost > state["capital"]:
        log.warning("Capital insuffisant.")
        return

    if PAPER_TRADING:
        log.info(f"[PAPER] ACHAT  {qty} BTC @ {price:.2f} USDC  (coût : {cost:.2f} USDC)")
    else:
        try:
            exchange.create_market_buy_order(SYMBOL, qty)
            log.info(f"[RÉEL]  ACHAT  {qty} BTC @ ~{price:.2f} USDC")
        except Exception as e:
            log.error(f"Erreur ordre achat : {e}")
            return

    state["position"]    = "long"
    state["entry_price"] = price
    state["quantity"]    = qty
    state["capital"]    -= cost
    state["trade_count"] += 1

def sell(exchange, price: float, reason: str = "signal") -> None:
    if state["position"] != "long":
        return

    qty     = state["quantity"]
    gain    = (price - state["entry_price"]) * qty
    pnl_pct = (price - state["entry_price"]) / state["entry_price"]

    if PAPER_TRADING:
        log.info(f"[PAPER] VENTE  {qty} BTC @ {price:.2f} USDC  PnL : {gain:+.2f} USDC ({pnl_pct:+.2%})  raison : {reason}")
    else:
        try:
            exchange.create_market_sell_order(SYMBOL, qty)
            log.info(f"[RÉEL]  VENTE  {qty} BTC @ ~{price:.2f} USDC  PnL : {gain:+.2f} USDC")
        except Exception as e:
            log.error(f"Erreur ordre vente : {e}")
            return

    state["capital"]    += qty * price
    state["pnl_total"]  += gain
    state["position"]    = None
    state["entry_price"] = 0.0
    state["quantity"]    = 0.0

# ─────────────────────────────────────────────
#  STOP-LOSS / TAKE-PROFIT
# ─────────────────────────────────────────────

def check_exit(exchange, price: float) -> None:
    if state["position"] != "long":
        return

    sl = state["entry_price"] * (1 - STOP_LOSS_PCT)
    tp = state["entry_price"] * (1 + TAKE_PROFIT_PCT)

    if price <= sl:
        log.warning(f"Stop-loss déclenché à {price:.2f} (seuil : {sl:.2f})")
        sell(exchange, price, reason="stop-loss")
    elif price >= tp:
        log.info(f"Take-profit déclenché à {price:.2f} (seuil : {tp:.2f})")
        sell(exchange, price, reason="take-profit")

# ─────────────────────────────────────────────
#  BOUCLE PRINCIPALE
# ─────────────────────────────────────────────

def run() -> None:
    mode = "PAPER TRADING" if PAPER_TRADING else "TRADING RÉEL ⚠️"
    log.info(f"═══ Démarrage du bot [{mode}] — {SYMBOL} ═══")

    exchange = connect()

    while True:
        try:
            if check_drawdown():
                break

            df    = get_ohlcv(exchange)
            df    = compute_indicators(df)
            price = get_price(exchange)

            rsi   = df.iloc[-1]["rsi"]
            ma50  = df.iloc[-1]["ma50"]
            ma200 = df.iloc[-1]["ma200"]

            log.info(
                f"Prix : {price:.2f} | RSI : {rsi:.1f} | "
                f"MA50 : {ma50:.2f} | MA200 : {ma200:.2f} | "
                f"Capital : {state['capital']:.2f} USDC | "
                f"PnL total : {state['pnl_total']:+.2f} USDC"
            )

            check_exit(exchange, price)

            sig = signal(df)
            log.info(f"Signal : {sig}")

            if sig == "BUY" and state["position"] is None:
                buy(exchange, price)
            elif sig == "SELL" and state["position"] == "long":
                sell(exchange, price, reason="signal")

        except Exception as e:
            log.error(f"Erreur : {e}", exc_info=True)

        time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    run()