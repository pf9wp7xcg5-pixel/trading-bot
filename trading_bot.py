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
    "capital_initial": None,  # sera défini au premier lancement
    "pnl_total"      : 0.0,
    "trade_count"    : 0,
}

# ─────────────────────────────────────────────
#  CONNEXION KRAKEN
# ─────────────────────────────────────────────

def connect():
    exchange = ccxt.kraken({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "timeout": 30000,
        "enableRateLimit": True,
    })
    log.info("Connexion Kraken établie.")
    return exchange

# ─────────────────────────────────────────────
#  SOLDE RÉEL
# ─────────────────────────────────────────────

def get_usdc_balance(exchange) -> float:
    """Récupère le solde USDC réel disponible sur Kraken."""
    if PAPER_TRADING:
        return state.get("paper_capital", state["capital_initial"] or 30.0)
    try:
        balance = exchange.fetch_balance()
        return float(balance["free"].get("USDC", 0.0))
    except Exception as e:
        log.error(f"Erreur récupération solde : {e}")
        return 0.0

def get_btc_balance(exchange) -> float:
    """Récupère le solde BTC réel disponible sur Kraken."""
    if PAPER_TRADING:
        return state.get("paper_btc", 0.0)
    try:
        balance = exchange.fetch_balance()
        return float(balance["free"].get("BTC", 0.0) or balance["free"].get("XBT", 0.0))
    except Exception as e:
        log.error(f"Erreur récupération solde BTC : {e}")
        return 0.0

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
    """Calcule la quantité à acheter selon le capital réel disponible."""
    risk_amount   = capital * RISK_PER_TRADE
    stop_distance = price * STOP_LOSS_PCT
    qty = risk_amount / stop_distance
    return round(qty, 6)

def check_drawdown(exchange, price: float) -> bool:
    """Calcule le drawdown sur la valeur totale réelle du portefeuille."""
    if state["capital_initial"] is None:
        return False
    usdc  = get_usdc_balance(exchange)
    btc   = get_btc_balance(exchange)
    total = usdc + btc * price
    loss_pct = (state["capital_initial"] - total) / state["capital_initial"]
    if loss_pct >= MAX_DRAWDOWN:
        log.warning(f"Drawdown maximum atteint ({loss_pct:.1%}). Arrêt du bot.")
        return True
    return False

# ─────────────────────────────────────────────
#  ORDRES
# ─────────────────────────────────────────────

def buy(exchange, price: float) -> None:
    capital = get_usdc_balance(exchange)
    if capital < 1.0:
        log.warning(f"Capital insuffisant : {capital:.2f} USDC")
        return

    qty  = position_size(capital, price)
    cost = qty * price

    # Sécurité : ne pas dépenser plus que le capital disponible
    if cost > capital:
        qty  = round((capital * 0.95) / price, 6)
        cost = qty * price

    if PAPER_TRADING:
        log.info(f"[PAPER] ACHAT  {qty} BTC @ {price:.2f} USDC  (coût : {cost:.2f} USDC)")
        state["paper_capital"] = capital - cost
        state["paper_btc"]     = state.get("paper_btc", 0.0) + qty
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
    state["trade_count"] += 1

def sell(exchange, price: float, reason: str = "signal") -> None:
    if state["position"] != "long":
        return

    qty     = state["quantity"]
    gain    = (price - state["entry_price"]) * qty
    pnl_pct = (price - state["entry_price"]) / state["entry_price"]

    if PAPER_TRADING:
        log.info(f"[PAPER] VENTE  {qty} BTC @ {price:.2f} USDC  PnL : {gain:+.2f} USDC ({pnl_pct:+.2%})  raison : {reason}")
        state["paper_capital"] = state.get("paper_capital", 0.0) + qty * price
        state["paper_btc"]     = max(0.0, state.get("paper_btc", 0.0) - qty)
    else:
        try:
            exchange.create_market_sell_order(SYMBOL, qty)
            log.info(f"[RÉEL]  VENTE  {qty} BTC @ ~{price:.2f} USDC  PnL : {gain:+.2f} USDC ({pnl_pct:+.2%})")
        except Exception as e:
            log.error(f"Erreur ordre vente : {e}")
            return

    state["pnl_total"] += gain
    state["position"]   = None
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

    # Initialise le capital initial depuis le solde réel
    usdc = get_usdc_balance(exchange)
    btc  = get_btc_balance(exchange)

    while True:
        try:
            df    = get_ohlcv(exchange)
            df    = compute_indicators(df)
            price = get_price(exchange)

            # Initialise le capital initial au premier cycle
            if state["capital_initial"] is None:
                state["capital_initial"] = usdc + btc * price
                if PAPER_TRADING:
                    state["paper_capital"] = usdc
                    state["paper_btc"]     = btc
                log.info(f"Capital initial : {state['capital_initial']:.2f} USDC")

            # Solde actuel
            usdc = get_usdc_balance(exchange)
            btc  = get_btc_balance(exchange)
            total = usdc + btc * price

            rsi   = df.iloc[-1]["rsi"]
            ma50  = df.iloc[-1]["ma50"]
            ma200 = df.iloc[-1]["ma200"]

            log.info(
                f"Prix : {price:.2f} | RSI : {rsi:.1f} | "
                f"MA50 : {ma50:.2f} | MA200 : {ma200:.2f} | "
                f"USDC : {usdc:.2f} | BTC : {btc:.6f} | "
                f"Total : {total:.2f} USDC | PnL : {state['pnl_total']:+.2f} USDC"
            )

            if check_drawdown(exchange, price):
                break

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