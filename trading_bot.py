"""
Bot de trading algorithmique — Stratégie RSI + Moyennes Mobiles
Exchange : Kraken (via ccxt)
Paires : BTC/USDC + ETH/USDC (multi-paires avec capital séparé)

Installation :
    pip install ccxt pandas ta
"""

import time
import json
import logging
import os
import ccxt
import pandas as pd
import ta

# ─────────────────────────────────────────────
#  CONFIGURATION GLOBALE
# ─────────────────────────────────────────────

API_KEY    = os.environ.get("KRAKEN_API_KEY", "")
API_SECRET = os.environ.get("KRAKEN_API_SECRET", "")

PAPER_TRADING = os.environ.get("PAPER_TRADING", "true").lower() == "true"
LOOP_INTERVAL = 60 * 60  # toutes les heures

# Fichier de persistance de l'état (positions, capital, PnL) entre redémarrages.
STATE_FILE = os.environ.get("STATE_FILE_PATH", "bot_state.json")

# Portefeuille simulé utilisé pour calculer l'allocation 70/30 en mode PAPER_TRADING
PAPER_TRADING_SIMULATED_TOTAL = float(os.environ.get("PAPER_TRADING_SIMULATED_TOTAL", "322.0"))

# Taux de frais Kraken (taker) appliqué à l'achat et à la vente, en proportion (0.008 = 0.8%).
# Basé sur le taux réellement observé sur l'historique de trades (ledger Kraken) depuis le
# 15/07 — ce taux dépend du palier de volume 30 jours et peut changer ; à ajuster si besoin.
TAKER_FEE_PCT = float(os.environ.get("TAKER_FEE_PCT", "0.008"))

# ─────────────────────────────────────────────
#  CONFIGURATION PAR PAIRE
# ─────────────────────────────────────────────

PAIRS_CONFIG = {
    "BTC/USDC": {
        "allocation_pct": 0.70,
        "risk_per_trade": 0.02,
        "stop_loss_pct":  0.03,
        "take_profit_pct":0.06,
        "max_drawdown":   0.10,
        "rsi_buy":        40,
        "rsi_sell":       65,
        "timeframe":      "1h",
        "cooldown_hours": 4,
    },
    "ETH/USDC": {
        "allocation_pct": 0.30,
        "risk_per_trade": 0.02,
        "stop_loss_pct":  0.04,
        "take_profit_pct":0.08,
        "max_drawdown":   0.10,
        "rsi_buy":        40,
        "rsi_sell":       65,
        "timeframe":      "1h",
        "cooldown_hours": 4,
    },
}

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
#  ÉTAT PAR PAIRE
# ─────────────────────────────────────────────

def init_state(config: dict) -> dict:
    return {
        "position":        None,
        "entry_price":     0.0,
        "quantity":        0.0,
        "cost_basis":      0.0,         # coût réel payé à l'achat (prix × qté + frais)
        "capital":         config["capital"],
        "capital_initial": config["capital"],
        "pnl_total":       0.0,
        "trade_count":     0,
        "suspended":       False,
        "cooldown_until":  None,
    }

def load_states(configs: dict):
    saved = {}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                saved = json.load(f)
        except Exception as e:
            log.error(f"Impossible de lire {STATE_FILE} ({e}) — initialisation à neuf.")
            saved = {}
    else:
        log.warning(
            f"Aucun fichier d'état trouvé ({STATE_FILE}) — initialisation à neuf. "
            f"Si le bot a déjà tradé avant aujourd'hui, vérifie que le Volume Railway "
            f"est bien monté, sinon l'historique de position est perdu."
        )

    states = {}
    fresh_symbols = []
    for sym in configs:
        if sym in saved:
            states[sym] = saved[sym]
            _migrate_state(states[sym])
            log.info(f"[{sym}] État restauré depuis {STATE_FILE} (position : {states[sym]['position']})")
        else:
            states[sym] = None
            fresh_symbols.append(sym)
    return states, fresh_symbols

def _migrate_state(state: dict) -> None:
    """
    Complète un état chargé depuis un ancien format de bot_state.json qui n'aurait pas
    encore le champ 'cost_basis' (introduit avec la prise en compte des frais Kraken).
    Si une position est ouverte sans cost_basis connu, on l'estime à partir du prix
    d'entrée enregistré + une estimation des frais, plutôt que de planter à la vente.
    """
    if "cost_basis" not in state:
        if state.get("position") == "long" and state.get("entry_price", 0) > 0:
            estimated = state["quantity"] * state["entry_price"] * (1 + TAKER_FEE_PCT)
            state["cost_basis"] = estimated
            log.warning(
                f"Ancien format d'état détecté (sans cost_basis) pour une position ouverte. "
                f"Estimation : {estimated:.2f} USDC (prix d'entrée × quantité × (1+frais)). "
                f"Le PnL de la prochaine vente sera approximatif, pas exact."
            )
        else:
            state["cost_basis"] = 0.0

def save_states(states: dict) -> None:
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(states, f, indent=2)
    except Exception as e:
        log.error(f"Impossible d'écrire {STATE_FILE} : {e}")

def check_startup_balance(exchange, states: dict) -> None:
    try:
        balance   = exchange.fetch_balance()
        usdc_free = float(balance.get("USDC", {}).get("free", 0.0))
    except Exception as e:
        log.error(f"Vérification du solde au démarrage impossible : {e}")
        return

    expected_cash = sum(s["capital"] for s in states.values())
    diff          = abs(usdc_free - expected_cash)
    tolerance     = max(5.0, 0.05 * expected_cash)

    if diff > tolerance:
        log.warning(
            f"⚠️ Écart détecté au démarrage : cash attendu par le bot = {expected_cash:.2f} USDC, "
            f"solde USDC réel sur Kraken = {usdc_free:.2f} USDC (écart : {diff:.2f} USDC). "
            f"Vérifie manuellement les positions avant de laisser le bot trader en mode réel."
        )
    else:
        log.info(f"Vérification du solde OK — attendu : {expected_cash:.2f} USDC, réel : {usdc_free:.2f} USDC.")

# ─────────────────────────────────────────────
#  CONNEXION
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
#  DONNÉES DE MARCHÉ
# ─────────────────────────────────────────────

def get_ohlcv(exchange, symbol: str, timeframe: str, limit: int = 250) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["close"] = df["close"].astype(float)
    return df

def get_price(exchange, symbol: str) -> float:
    return float(exchange.fetch_ticker(symbol)["last"])

# ─────────────────────────────────────────────
#  RÉCONCILIATION AU DÉMARRAGE
# ─────────────────────────────────────────────

DUST_THRESHOLD = {"BTC": 0.0001, "ETH": 0.001}

# ─────────────────────────────────────────────
#  ALLOCATION DYNAMIQUE DU CAPITAL
# ─────────────────────────────────────────────

def total_portfolio_value(exchange, symbols) -> float:
    balance = exchange.fetch_balance()
    total = float(balance.get("USDC", {}).get("total", 0.0))
    for symbol in symbols:
        base = symbol.split("/")[0]
        qty = float(balance.get(base, {}).get("total", 0.0))
        if qty > DUST_THRESHOLD.get(base, 0.0):
            total += qty * get_price(exchange, symbol)
    return total

def allocate_capital(exchange, configs: dict, fresh_symbols: list) -> None:
    if not fresh_symbols:
        return

    if PAPER_TRADING:
        total_value = PAPER_TRADING_SIMULATED_TOTAL
        log.info(f"[PAPER] Portefeuille simulé utilisé pour l'allocation : {total_value:.2f} USDC")
    else:
        total_value = total_portfolio_value(exchange, configs.keys())
        log.info(f"Valeur totale réelle du portefeuille (cash + positions) : {total_value:.2f} USDC")

    for sym in fresh_symbols:
        cfg = configs[sym]
        cfg["capital"] = round(total_value * cfg["allocation_pct"], 2)
        log.info(
            f"[{sym}] Capital initial alloué dynamiquement : {cfg['capital']:.2f} USDC "
            f"({cfg['allocation_pct']*100:.0f}% de {total_value:.2f} USDC)"
        )


def reconcile_startup_position(exchange, symbol: str, state: dict, cfg: dict) -> None:
    if state["position"] is not None:
        return

    base_asset = symbol.split("/")[0]
    try:
        balance = exchange.fetch_balance()
        qty     = float(balance.get(base_asset, {}).get("free", 0.0))
    except Exception as e:
        log.error(f"[{symbol}] Impossible de vérifier le solde {base_asset} pour réconciliation : {e}")
        return

    if qty <= DUST_THRESHOLD.get(base_asset, 0.0):
        return

    entry_price = None
    real_fees   = 0.0
    try:
        trades = exchange.fetch_my_trades(symbol, limit=50)
        buys   = [t for t in trades if t.get("side") == "buy"]
        if buys:
            total_cost = sum(t["price"] * t["amount"] for t in buys)
            total_qty  = sum(t["amount"] for t in buys)
            # Frais réels payés à l'achat, quand l'exchange les renvoie dans le trade
            # (utilisés pour reconstruire un cost_basis fidèle, sinon on retombe sur
            # une estimation via TAKER_FEE_PCT juste en dessous).
            real_fees = sum(
                t["fee"]["cost"] for t in buys
                if t.get("fee") and t["fee"].get("currency") in (None, "USDC")
            )
            if total_qty > 0:
                entry_price = total_cost / total_qty

    except Exception as e:
        log.error(f"[{symbol}] Impossible de lire l'historique de trades : {e}")

    if entry_price is None:
        entry_price = get_price(exchange, symbol)
        log.warning(
            f"[{symbol}] Position détectée ({qty} {base_asset}) mais aucun historique de "
            f"trades exploitable. Prix d'entrée approximé au prix actuel ({entry_price:.2f}) "
            f"— le stop-loss/take-profit démarre à partir de maintenant, pas du vrai prix d'achat."
        )

    invested = qty * entry_price
    # Si on n'a pas pu récupérer les frais réels (ex : entry_price approximé), on les
    # estime avec le taux configuré pour ne pas sous-évaluer le coût réel de la position.
    fees_component = real_fees if real_fees > 0 else invested * TAKER_FEE_PCT
    cost_basis = invested + fees_component

    state["position"]    = "long"
    state["quantity"]    = qty
    state["entry_price"] = entry_price
    state["cost_basis"]  = cost_basis
    state["capital"]     = max(0.0, cfg["capital"] - cost_basis)

    log.warning(
        f"[{symbol}] Position reconstituée automatiquement : {qty} {base_asset} @ {entry_price:.2f} "
        f"(cash restant estimé pour cette paire : {state['capital']:.2f} USDC). "
        f"Vérifie que ça correspond bien à la réalité de ton portefeuille."
    )

# ─────────────────────────────────────────────
#  INDICATEURS
# ─────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["rsi"]   = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["ma50"]  = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()
    return df

# ─────────────────────────────────────────────
#  SIGNAL
# ─────────────────────────────────────────────

def signal(df: pd.DataFrame, cfg: dict) -> str:
    last = df.iloc[-1]
    prev = df.iloc[-2]

    bullish = last["ma50"] > last["ma200"]
    bearish = last["ma50"] < last["ma200"]

    golden_cross = prev["ma50"] <= prev["ma200"] and last["ma50"] > last["ma200"]
    death_cross  = prev["ma50"] >= prev["ma200"] and last["ma50"] < last["ma200"]

    rsi_oversold     = last["rsi"] < cfg["rsi_buy"]
    rsi_overbought   = last["rsi"] > cfg["rsi_sell"]
    rsi_crosses_up   = prev["rsi"] < cfg["rsi_buy"]  and last["rsi"] >= cfg["rsi_buy"]
    rsi_crosses_down = prev["rsi"] > cfg["rsi_sell"] and last["rsi"] <= cfg["rsi_sell"]

    if golden_cross:
        return "BUY"
    if death_cross:
        return "SELL"
    if (rsi_oversold or rsi_crosses_up) and bullish:
        return "BUY"
    if (rsi_overbought or rsi_crosses_down) and bearish:
        return "SELL"
    return "HOLD"

# ─────────────────────────────────────────────
#  GESTION DU RISQUE
# ─────────────────────────────────────────────

def position_size(capital: float, price: float, cfg: dict) -> float:
    risk    = capital * cfg["risk_per_trade"]
    stop_d  = price * cfg["stop_loss_pct"]
    return round(risk / stop_d, 6)

def equity(state: dict, price: float) -> float:
    position_value = state["quantity"] * price if state["position"] == "long" else 0.0
    return state["capital"] + position_value

def check_drawdown(symbol: str, state: dict, cfg: dict, price: float) -> bool:
    eq   = equity(state, price)
    loss = (state["capital_initial"] - eq) / state["capital_initial"]

    if loss >= cfg["max_drawdown"]:
        if not state["suspended"]:
            log.warning(
                f"[{symbol}] Drawdown max atteint ({loss:.1%}, équité : {eq:.2f} USDC). "
                f"Paire suspendue."
            )
            state["suspended"] = True
        return True

    if state["suspended"]:
        log.info(f"[{symbol}] Équité remontée au-dessus du seuil ({loss:.1%}). Paire réactivée.")
    state["suspended"] = False
    return False

# ─────────────────────────────────────────────
#  ORDRES
# ─────────────────────────────────────────────

def buy(exchange, symbol: str, price: float, state: dict, cfg: dict) -> None:
    capital = state["capital"]
    if capital < 1.0:
        log.warning(f"[{symbol}] Capital insuffisant : {capital:.2f} USDC")
        return

    qty       = position_size(capital, price, cfg)
    cost      = qty * price
    fee       = cost * TAKER_FEE_PCT
    total_cost = cost + fee
    if total_cost > capital:
        # On laisse une petite marge (0.95) pour rester sûr de couvrir prix + frais
        qty        = round((capital * 0.95) / (price * (1 + TAKER_FEE_PCT)), 6)
        cost       = qty * price
        fee        = cost * TAKER_FEE_PCT
        total_cost = cost + fee

    if PAPER_TRADING:
        log.info(f"[PAPER][{symbol}] ACHAT {qty} @ {price:.2f} (coût : {cost:.2f} + frais {fee:.2f} = {total_cost:.2f} USDC)")
    else:
        try:
            exchange.create_market_buy_order(symbol, qty)
            log.info(f"[RÉEL] [{symbol}] ACHAT {qty} @ ~{price:.2f} USDC (frais estimés : {fee:.2f} USDC)")
        except Exception as e:
            log.error(f"[{symbol}] Erreur achat : {e}")
            return

    state["position"]    = "long"
    state["entry_price"] = price          # prix de marché pur, sert de référence pour SL/TP
    state["quantity"]    = qty
    state["cost_basis"]  = total_cost     # coût réel payé (frais inclus), sert au calcul du PnL
    state["capital"]    -= total_cost
    state["trade_count"] += 1

def sell(exchange, symbol: str, price: float, state: dict, reason: str = "signal", cfg: dict = None) -> None:
    if state["position"] != "long":
        return

    qty           = state["quantity"]
    proceeds      = qty * price
    fee           = proceeds * TAKER_FEE_PCT
    net_proceeds  = proceeds - fee
    gain          = net_proceeds - state["cost_basis"]  # PnL net, frais d'achat ET de vente inclus
    pnl_pct       = gain / state["cost_basis"] if state["cost_basis"] else 0.0

    if PAPER_TRADING:
        log.info(f"[PAPER][{symbol}] VENTE {qty} @ {price:.2f}  PnL net : {gain:+.2f} USDC ({pnl_pct:+.2%})  raison : {reason}")
    else:
        try:
            exchange.create_market_sell_order(symbol, qty)
            log.info(f"[RÉEL] [{symbol}] VENTE {qty} @ ~{price:.2f}  PnL net : {gain:+.2f} USDC ({pnl_pct:+.2%})  (frais estimés : {fee:.2f} USDC)")
        except Exception as e:
            log.error(f"[{symbol}] Erreur vente : {e}")
            return

    state["capital"]    += net_proceeds
    state["pnl_total"]  += gain
    state["position"]    = None
    state["entry_price"] = 0.0
    state["quantity"]    = 0.0
    state["cost_basis"]  = 0.0

    if reason == "stop-loss" and cfg is not None:
        hours = cfg.get("cooldown_hours", 4)
        state["cooldown_until"] = time.time() + hours * 3600
        log.warning(
            f"[{symbol}] Cooldown activé après stop-loss : pas de nouvel achat avant "
            f"{hours}h (jusqu'à {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(state['cooldown_until']))})."
        )

def in_cooldown(state: dict) -> bool:
    cooldown_until = state.get("cooldown_until")
    return cooldown_until is not None and time.time() < cooldown_until

# ─────────────────────────────────────────────
#  STOP-LOSS / TAKE-PROFIT
# ─────────────────────────────────────────────

def check_exit(exchange, symbol: str, price: float, state: dict, cfg: dict) -> None:
    if state["position"] != "long":
        return

    sl = state["entry_price"] * (1 - cfg["stop_loss_pct"])
    tp = state["entry_price"] * (1 + cfg["take_profit_pct"])

    if price <= sl:
        log.warning(f"[{symbol}] Stop-loss déclenché à {price:.2f} (seuil : {sl:.2f})")
        sell(exchange, symbol, price, state, reason="stop-loss", cfg=cfg)
    elif price >= tp:
        log.info(f"[{symbol}] Take-profit déclenché à {price:.2f} (seuil : {tp:.2f})")
        sell(exchange, symbol, price, state, reason="take-profit", cfg=cfg)

# ─────────────────────────────────────────────
#  BOUCLE PRINCIPALE
# ─────────────────────────────────────────────

def run() -> None:
    mode = "PAPER TRADING" if PAPER_TRADING else "TRADING RÉEL ⚠️"
    log.info(f"═══ Démarrage du bot multi-paires [{mode}] ═══")
    for sym, cfg in PAIRS_CONFIG.items():
        log.info(f"  {sym} → allocation : {cfg['allocation_pct']*100:.0f}% | SL : {cfg['stop_loss_pct']*100:.0f}% | TP : {cfg['take_profit_pct']*100:.0f}%")

    exchange = connect()

    states, fresh = load_states(PAIRS_CONFIG)

    allocate_capital(exchange, PAIRS_CONFIG, fresh)
    for sym in fresh:
        states[sym] = init_state(PAIRS_CONFIG[sym])

    if not PAPER_TRADING:
        for sym, cfg in PAIRS_CONFIG.items():
            reconcile_startup_position(exchange, sym, states[sym], cfg)
        check_startup_balance(exchange, states)
        save_states(states)

    while True:
        for symbol, cfg in PAIRS_CONFIG.items():
            state = states[symbol]
            try:
                price = get_price(exchange, symbol)

                check_exit(exchange, symbol, price, state, cfg)

                suspended = check_drawdown(symbol, state, cfg, price)
                if suspended:
                    log.info(
                        f"[{symbol}] Suspendue (drawdown) — nouvelles entrées bloquées, "
                        f"stop-loss/take-profit toujours actifs."
                    )
                    continue

                df    = get_ohlcv(exchange, symbol, cfg["timeframe"])
                df    = compute_indicators(df)

                rsi   = df.iloc[-1]["rsi"]
                ma50  = df.iloc[-1]["ma50"]
                ma200 = df.iloc[-1]["ma200"]

                log.info(
                    f"[{symbol}] Prix : {price:.2f} | RSI : {rsi:.1f} | "
                    f"MA50 : {ma50:.2f} | MA200 : {ma200:.2f} | "
                    f"Capital : {state['capital']:.2f} USDC | "
                    f"Équité : {equity(state, price):.2f} USDC | "
                    f"PnL : {state['pnl_total']:+.2f} USDC"
                )

                sig = signal(df, cfg)
                log.info(f"[{symbol}] Signal : {sig}")

                if sig == "BUY" and state["position"] is None:
                    if in_cooldown(state):
                        remaining_min = (state["cooldown_until"] - time.time()) / 60
                        log.info(
                            f"[{symbol}] Signal BUY ignoré — cooldown actif encore "
                            f"{remaining_min:.0f} min après le dernier stop-loss."
                        )
                    else:
                        buy(exchange, symbol, price, state, cfg)
                elif sig == "SELL" and state["position"] == "long":
                    sell(exchange, symbol, price, state, reason="signal", cfg=cfg)

            except Exception as e:
                log.error(f"[{symbol}] Erreur : {e}", exc_info=True)

            save_states(states)
            time.sleep(2)

        time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    run()