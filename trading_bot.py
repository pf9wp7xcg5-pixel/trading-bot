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
# ⚠️ Sur Railway, le système de fichiers est éphémère par défaut : il faut
# attacher un "Volume" au service et le monter sur ce chemin (ex : /data)
# pour que l'état survive à un redéploiement. Sinon ce fichier repart de zéro
# à chaque déploiement, exactement comme le run du 15 juillet.
STATE_FILE = os.environ.get("STATE_FILE_PATH", "bot_state.json")

# Portefeuille simulé utilisé pour calculer l'allocation 70/30 en mode PAPER_TRADING
# (pas de vrai solde Kraken à interroger dans ce cas).
PAPER_TRADING_SIMULATED_TOTAL = float(os.environ.get("PAPER_TRADING_SIMULATED_TOTAL", "322.0"))

# ─────────────────────────────────────────────
#  CONFIGURATION PAR PAIRE
# ─────────────────────────────────────────────

PAIRS_CONFIG = {
    "BTC/USDC": {
        "allocation_pct": 0.70,   # 70% du portefeuille total (cash + positions)
        "risk_per_trade": 0.02,
        "stop_loss_pct":  0.03,
        "take_profit_pct":0.06,
        "max_drawdown":   0.10,
        "rsi_buy":        40,
        "rsi_sell":       65,
        "timeframe":      "1h",
        "cooldown_hours": 4,      # pas de rachat avant 4h après un stop-loss
    },
    "ETH/USDC": {
        "allocation_pct": 0.30,   # 30% du portefeuille total (cash + positions)
        "risk_per_trade": 0.02,
        "stop_loss_pct":  0.04,    # ETH plus volatile → stop plus large
        "take_profit_pct":0.08,    # ETH plus volatile → TP plus ambitieux
        "max_drawdown":   0.10,
        "rsi_buy":        40,
        "rsi_sell":       65,
        "timeframe":      "1h",
        "cooldown_hours": 4,      # pas de rachat avant 4h après un stop-loss
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
        "capital":         config["capital"],
        "capital_initial": config["capital"],
        "pnl_total":       0.0,
        "trade_count":     0,
        "suspended":       False,       # True si le drawdown max a été atteint
        "cooldown_until":  None,        # timestamp Unix : pas de rachat avant cette date suite à un stop-loss
    }

def load_states(configs: dict):
    """
    Recharge l'état sauvegardé si présent. Renvoie (states, fresh_symbols) où
    fresh_symbols est la liste des paires sans état persistant — celles-ci
    n'ont pas encore de "capital" défini et doivent passer par l'allocation
    dynamique (voir total_portfolio_value / répartition 70/30) avant init_state.
    """
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
            log.info(f"[{sym}] État restauré depuis {STATE_FILE} (position : {states[sym]['position']})")
        else:
            states[sym] = None  # sera initialisé après allocation dynamique du capital
            fresh_symbols.append(sym)
    return states, fresh_symbols

def save_states(states: dict) -> None:
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(states, f, indent=2)
    except Exception as e:
        log.error(f"Impossible d'écrire {STATE_FILE} : {e}")

def check_startup_balance(exchange, states: dict) -> None:
    """
    Compare le cash que le bot croit détenir (somme des 'capital' en mémoire/fichier)
    au solde USDC réel sur Kraken. Un écart important signale un état désynchronisé
    (ex : redéploiement sans persistance, trade manuel sur le compte, etc.).
    """
    try:
        balance   = exchange.fetch_balance()
        usdc_free = float(balance.get("USDC", {}).get("free", 0.0))
    except Exception as e:
        log.error(f"Vérification du solde au démarrage impossible : {e}")
        return

    expected_cash = sum(s["capital"] for s in states.values())
    diff          = abs(usdc_free - expected_cash)
    tolerance     = max(5.0, 0.05 * expected_cash)  # 5 USDC ou 5%, le plus grand des deux

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

# En dessous de ces quantités, on considère que c'est de la poussière
# (reliquat d'arrondi/frais) et pas une vraie position.
DUST_THRESHOLD = {"BTC": 0.0001, "ETH": 0.001}

# ─────────────────────────────────────────────
#  ALLOCATION DYNAMIQUE DU CAPITAL
# ─────────────────────────────────────────────

def total_portfolio_value(exchange, symbols) -> float:
    """Valeur totale réelle du portefeuille (cash USDC + toutes les positions détenues)."""
    balance = exchange.fetch_balance()
    total = float(balance.get("USDC", {}).get("total", 0.0))
    for symbol in symbols:
        base = symbol.split("/")[0]
        qty = float(balance.get(base, {}).get("total", 0.0))
        if qty > DUST_THRESHOLD.get(base, 0.0):
            total += qty * get_price(exchange, symbol)
    return total

def allocate_capital(exchange, configs: dict, fresh_symbols: list) -> None:
    """
    Détermine dynamiquement le 'capital' (au sens PAIRS_CONFIG) des paires qui n'ont
    pas encore d'état persistant, en répartissant la valeur totale actuelle du
    portefeuille selon 'allocation_pct'. Une fois l'état sauvegardé, ce calcul n'est
    plus refait pour cette paire — seul le premier lancement (ou l'ajout d'une
    nouvelle paire) déclenche une nouvelle allocation.
    """
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
    """
    Si le bot n'a aucune position en mémoire/fichier mais que le compte Kraken détient
    réellement l'actif (ex : achat fait avant la mise en place de la persistance, ou état
    perdu suite à un redéploiement), on reconstruit la position à partir de l'historique
    de trades réel plutôt que de l'ignorer silencieusement.
    """
    if state["position"] is not None:
        return  # position déjà connue via le fichier d'état, rien à faire

    base_asset = symbol.split("/")[0]
    try:
        balance = exchange.fetch_balance()
        qty     = float(balance.get(base_asset, {}).get("free", 0.0))
    except Exception as e:
        log.error(f"[{symbol}] Impossible de vérifier le solde {base_asset} pour réconciliation : {e}")
        return

    if qty <= DUST_THRESHOLD.get(base_asset, 0.0):
        return  # rien détenu de significatif, l'état "aucune position" est correct

    # Position détectée mais inconnue du bot → on cherche le prix d'entrée réel
    entry_price = None
    try:
        trades = exchange.fetch_my_trades(symbol, limit=50)
        buys   = [t for t in trades if t.get("side") == "buy"]
        if buys:
            total_cost = sum(t["price"] * t["amount"] for t in buys)
            total_qty  = sum(t["amount"] for t in buys)
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
    state["position"]    = "long"
    state["quantity"]    = qty
    state["entry_price"] = entry_price
    state["capital"]     = max(0.0, cfg["capital"] - invested)

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

    # Croisement de moyennes mobiles (signal de suivi de tendance, complémentaire au RSI).
    # Golden cross : MA50 repasse au-dessus de MA200 → la tendance redevient haussière.
    # Death cross  : MA50 repasse en dessous de MA200 → la tendance redevient baissière.
    # Sans ça, le bot ne rentre/sort que sur des extrêmes RSI et peut rester à l'écart
    # de rallyes qui montent sans jamais repasser en survente (ou de chutes qui
    # continuent sans repasser en surachat).
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
    """Valeur totale = cash disponible + valeur de la position ouverte au prix actuel."""
    position_value = state["quantity"] * price if state["position"] == "long" else 0.0
    return state["capital"] + position_value

def check_drawdown(symbol: str, state: dict, cfg: dict, price: float) -> bool:
    """
    Calcule le drawdown sur l'équité totale (cash + position ouverte), pas seulement
    le cash restant — sinon un simple achat est compté comme une perte.
    """
    eq   = equity(state, price)
    loss = (state["capital_initial"] - eq) / state["capital_initial"]

    if loss >= cfg["max_drawdown"]:
        if not state["suspended"]:
            # On ne log l'alerte qu'une seule fois, au moment où elle se déclenche,
            # pour ne pas spammer les logs à chaque boucle (toutes les heures).
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

    qty  = position_size(capital, price, cfg)
    cost = qty * price
    if cost > capital:
        qty  = round((capital * 0.95) / price, 6)
        cost = qty * price

    if PAPER_TRADING:
        log.info(f"[PAPER][{symbol}] ACHAT {qty} @ {price:.2f} (coût : {cost:.2f} USDC)")
    else:
        try:
            exchange.create_market_buy_order(symbol, qty)
            log.info(f"[RÉEL] [{symbol}] ACHAT {qty} @ ~{price:.2f} USDC")
        except Exception as e:
            log.error(f"[{symbol}] Erreur achat : {e}")
            return

    state["position"]    = "long"
    state["entry_price"] = price
    state["quantity"]    = qty
    state["capital"]    -= cost
    state["trade_count"] += 1

def sell(exchange, symbol: str, price: float, state: dict, reason: str = "signal", cfg: dict = None) -> None:
    if state["position"] != "long":
        return

    qty     = state["quantity"]
    gain    = (price - state["entry_price"]) * qty
    pnl_pct = (price - state["entry_price"]) / state["entry_price"]

    if PAPER_TRADING:
        log.info(f"[PAPER][{symbol}] VENTE {qty} @ {price:.2f}  PnL : {gain:+.2f} USDC ({pnl_pct:+.2%})  raison : {reason}")
    else:
        try:
            exchange.create_market_sell_order(symbol, qty)
            log.info(f"[RÉEL] [{symbol}] VENTE {qty} @ ~{price:.2f}  PnL : {gain:+.2f} USDC ({pnl_pct:+.2%})")
        except Exception as e:
            log.error(f"[{symbol}] Erreur vente : {e}")
            return

    state["capital"]    += qty * price
    state["pnl_total"]  += gain
    state["position"]    = None
    state["entry_price"] = 0.0
    state["quantity"]    = 0.0

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

    # Recharger l'état sauvegardé (positions, capital, PnL) s'il existe.
    # 'fresh' = paires sans état persistant, qui n'ont pas encore de capital défini.
    states, fresh = load_states(PAIRS_CONFIG)

    # Allocation dynamique 70/30 (ou selon 'allocation_pct') uniquement pour les
    # paires fraîches — une paire déjà tradée garde le capital fixé lors de son
    # tout premier lancement, mis à jour ensuite par les achats/ventes réels.
    allocate_capital(exchange, PAIRS_CONFIG, fresh)
    for sym in fresh:
        states[sym] = init_state(PAIRS_CONFIG[sym])

    if not PAPER_TRADING:
        for sym, cfg in PAIRS_CONFIG.items():
            reconcile_startup_position(exchange, sym, states[sym], cfg)
        check_startup_balance(exchange, states)
        save_states(states)  # persiste immédiatement le résultat de l'allocation/réconciliation

    while True:
        for symbol, cfg in PAIRS_CONFIG.items():
            state = states[symbol]
            try:
                price = get_price(exchange, symbol)

                # Le stop-loss / take-profit doit TOUJOURS pouvoir s'exécuter,
                # même si la paire est "suspendue" — sinon une position ouverte
                # reste sans protection en cas de retournement du marché.
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

            save_states(states)  # persiste l'état à chaque itération, même en cas d'erreur
            time.sleep(2)  # petite pause entre les deux paires

        time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    run()