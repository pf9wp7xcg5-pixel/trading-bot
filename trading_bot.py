"""
Bot de trading algorithmique — Stratégie RSI + Moyennes Mobiles
Broker : Binance (via python-binance)
Mode : Paper Trading activé par défaut (aucun ordre réel)

Installation :
    pip install python-binance pandas ta

Configuration :
    Renseigne ta clé API et secrète Binance dans les variables en haut du fichier.
    Pour rester en mode simulation, laisse PAPER_TRADING = True.
"""

import time
import logging
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import ta

# ip de railway ? 
import requests
ip = requests.get("https://api.ipify.org").text
print(f"IP Railway : {ip}")


# ─────────────────────────────────────────────
#  CONFIGURATION — à modifier
# ─────────────────────────────────────────────

import os
API_KEY    = os.environ.get("BINANCE_API_KEY", "vwLx2jvN8nhDJlD4MpPRxHeaM9yRC2QQRy1sSkoz6TqOHEBG9ao2bw4jN3j6oIzH")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "7wFL6yjSF0dqPKfuhHIsF3gJoVhRPvNHYXNDbSafTEHMAdDADluFVKgZvxS5XjRO")

SYMBOL          = "BTCUSDC"       # Paire tradée
INTERVAL        = Client.KLINE_INTERVAL_1HOUR  # Timeframe : 1h
CAPITAL_USDT    = 32.0            # Capital alloué en USDT
RISK_PER_TRADE  = 0.02            # Risque max par trade : 2 % du capital
STOP_LOSS_PCT   = 0.03            # Stop-loss : -3 %
TAKE_PROFIT_PCT = 0.06            # Take-profit : +6 %
MAX_DRAWDOWN    = 0.10            # Arrêt du bot si -10 % du capital initial

PAPER_TRADING   = False            # True = simulation | False = ordres réels
LOOP_INTERVAL   = 60 * 60          # Vérification toutes les heures (en secondes)

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
#  ÉTAT INTERNE DU BOT
# ─────────────────────────────────────────────

state = {
    "position"       : None,    # None | "long"
    "entry_price"    : 0.0,
    "quantity"       : 0.0,
    "capital"        : CAPITAL_USDT,
    "capital_initial": CAPITAL_USDT,
    "trade_count"    : 0,
    "pnl_total"      : 0.0,
}

# ─────────────────────────────────────────────
#  CONNEXION BINANCE
# ─────────────────────────────────────────────

def connect() -> Client:
    client = Client(API_KEY, API_SECRET, requests_params={"timeout": 30})
    log.info("Connexion Binance établie.")
    return client

# ─────────────────────────────────────────────
#  DONNÉES DE MARCHÉ
# ─────────────────────────────────────────────

def get_ohlcv(client: Client, symbol: str, interval: str, limit: int = 250) -> pd.DataFrame:
    """Récupère les bougies OHLCV et retourne un DataFrame."""
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades", "taker_base", "taker_quote", "ignore",
    ])
    df["close"] = df["close"].astype(float)
    df["high"]  = df["high"].astype(float)
    df["low"]   = df["low"].astype(float)
    df["open"]  = df["open"].astype(float)
    return df

def get_price(client: Client, symbol: str) -> float:
    return float(client.get_symbol_ticker(symbol=symbol)["price"])

# ─────────────────────────────────────────────
#  CALCUL DES INDICATEURS
# ─────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute RSI, MA50 et MA200 au DataFrame."""
    df["rsi"]   = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["ma50"]  = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()
    return df

# ─────────────────────────────────────────────
#  SIGNAUX
# ─────────────────────────────────────────────

def signal(df: pd.DataFrame) -> str:
    """
    Signal d'achat  : RSI < 35  ET  MA50 > MA200  (tendance haussière)
    Signal de vente : RSI > 65  ET  MA50 < MA200  (tendance baissière)
    """
    last = df.iloc[-1]
    prev = df.iloc[-2]

    bullish_trend = last["ma50"] > last["ma200"]
    bearish_trend = last["ma50"] < last["ma200"]

    rsi_oversold  = last["rsi"] < 40
    rsi_overbought = last["rsi"] > 65

    # On vérifie aussi un croisement RSI (plutôt qu'un niveau statique)
    rsi_crosses_up   = prev["rsi"] < 40 and last["rsi"] >= 40
    rsi_crosses_down = prev["rsi"] > 65 and last["rsi"] <= 65

    if (rsi_oversold or rsi_crosses_up) and bullish_trend:
        return "BUY"
    if (rsi_overbought or rsi_crosses_down) and bearish_trend:
        return "SELL"
    return "HOLD"

# ─────────────────────────────────────────────
#  GESTION DU RISQUE
# ─────────────────────────────────────────────

def position_size(capital: float, price: float) -> float:
    """Calcule la quantité à acheter en fonction du risque par trade."""
    risk_amount = capital * RISK_PER_TRADE
    stop_distance = price * STOP_LOSS_PCT
    qty = risk_amount / stop_distance
    # Arrondi à 5 décimales (précision BTC sur Binance)
    return round(qty, 5)

def check_drawdown() -> bool:
    """Retourne True si le drawdown maximum est atteint (arrêt du bot)."""
    loss_pct = (state["capital_initial"] - state["capital"]) / state["capital_initial"]
    if loss_pct >= MAX_DRAWDOWN:
        log.warning(f"Drawdown maximum atteint ({loss_pct:.1%}). Arrêt du bot.")
        return True
    return False

# ─────────────────────────────────────────────
#  EXÉCUTION DES ORDRES
# ─────────────────────────────────────────────

def buy(client: Client, price: float) -> None:
    qty = position_size(state["capital"], price)
    cost = qty * price

    if cost > state["capital"]:
        log.warning("Capital insuffisant pour cet achat.")
        return

    if PAPER_TRADING:
        log.info(f"[PAPER] ACHAT  {qty} {SYMBOL} @ {price:.2f} USDT  (coût : {cost:.2f} USDT)")
    else:
        try:
            client.order_market_buy(symbol=SYMBOL, quantity=qty)
            log.info(f"[RÉEL]  ACHAT  {qty} {SYMBOL} @ ~{price:.2f} USDT")
        except BinanceAPIException as e:
            log.error(f"Erreur ordre achat : {e}")
            return

    state["position"]    = "long"
    state["entry_price"] = price
    state["quantity"]    = qty
    state["capital"]    -= cost
    state["trade_count"] += 1

def sell(client: Client, price: float, reason: str = "signal") -> None:
    if state["position"] != "long":
        return

    qty  = state["quantity"]
    gain = (price - state["entry_price"]) * qty
    pnl_pct = (price - state["entry_price"]) / state["entry_price"]

    if PAPER_TRADING:
        log.info(f"[PAPER] VENTE  {qty} {SYMBOL} @ {price:.2f} USDT  PnL : {gain:+.2f} USDT ({pnl_pct:+.2%})  raison : {reason}")
    else:
        try:
            client.order_market_sell(symbol=SYMBOL, quantity=qty)
            log.info(f"[RÉEL]  VENTE  {qty} {SYMBOL} @ ~{price:.2f} USDT  PnL : {gain:+.2f} USDT")
        except BinanceAPIException as e:
            log.error(f"Erreur ordre vente : {e}")
            return

    state["capital"]   += qty * price
    state["pnl_total"] += gain
    state["position"]   = None
    state["entry_price"] = 0.0
    state["quantity"]    = 0.0

# ─────────────────────────────────────────────
#  STOP-LOSS / TAKE-PROFIT
# ─────────────────────────────────────────────

def check_exit(client: Client, price: float) -> None:
    """Vérifie stop-loss et take-profit sur la position ouverte."""
    if state["position"] != "long":
        return

    entry = state["entry_price"]
    sl    = entry * (1 - STOP_LOSS_PCT)
    tp    = entry * (1 + TAKE_PROFIT_PCT)

    if price <= sl:
        log.warning(f"Stop-loss déclenché à {price:.2f} USDT (seuil : {sl:.2f})")
        sell(client, price, reason="stop-loss")
    elif price >= tp:
        log.info(f"Take-profit déclenché à {price:.2f} USDT (seuil : {tp:.2f})")
        sell(client, price, reason="take-profit")

# ─────────────────────────────────────────────
#  BOUCLE PRINCIPALE
# ─────────────────────────────────────────────

def run() -> None:
    mode = "PAPER TRADING" if PAPER_TRADING else "TRADING RÉEL ⚠️"
    log.info(f"═══ Démarrage du bot [{mode}] — {SYMBOL} ═══")

    client = connect()

    while True:
        try:
            # 1. Vérification du drawdown
            if check_drawdown():
                break

            # 2. Données + indicateurs
            df    = get_ohlcv(client, SYMBOL, INTERVAL)
            df    = compute_indicators(df)
            price = get_price(client, SYMBOL)

            rsi   = df.iloc[-1]["rsi"]
            ma50  = df.iloc[-1]["ma50"]
            ma200 = df.iloc[-1]["ma200"]

            log.info(
                f"Prix : {price:.2f} | RSI : {rsi:.1f} | "
                f"MA50 : {ma50:.2f} | MA200 : {ma200:.2f} | "
                f"Capital : {state['capital']:.2f} USDT | "
                f"PnL total : {state['pnl_total']:+.2f} USDT"
            )

            # 3. Vérification stop-loss / take-profit
            check_exit(client, price)

            # 4. Signal
            sig = signal(df)
            log.info(f"Signal : {sig}")

            if sig == "BUY" and state["position"] is None:
                buy(client, price)
            elif sig == "SELL" and state["position"] == "long":
                sell(client, price, reason="signal")

        except BinanceAPIException as e:
            log.error(f"Erreur API Binance : {e}")
        except Exception as e:
            log.error(f"Erreur inattendue : {e}", exc_info=True)

        time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    run()
