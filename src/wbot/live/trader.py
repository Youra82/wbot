# src/wbot/live/trader.py
# Echtgeld auf Bitget: Market-Einstieg mit angehaengtem SL/TP (gleiche Abstaende wie trade_rules),
# Zeit-Ausstieg nach MAX_DAYS Tagen. Nur aktiv bei mode == "live" (siehe master_runner.py).
import os
import json
import logging
import pandas as pd

from wbot.forecast.trade_rules import levels, MAX_DAYS, SL_ATR

logger = logging.getLogger(__name__)
LIVE_STATE = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')),
                          'artifacts', 'live', 'live_state.json')


def symbol(coin):
    return f'{coin}/USDT:USDT'


def load_state():
    if os.path.exists(LIVE_STATE):
        with open(LIVE_STATE, encoding='utf-8') as f:
            return json.load(f)
    return {'open': {}, 'closed': []}


def save_state(state):
    os.makedirs(os.path.dirname(LIVE_STATE), exist_ok=True)
    with open(LIVE_STATE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _side_of(p):
    return 'long' if p.get('side') == 'long' else 'short'


def manage(exchange, state, today):
    """Gespeicherte Positionen mit der Boerse abgleichen; Zeit-Ausstieg. Rueckgabe: Meldungen."""
    msgs = []
    positions = exchange.fetch_open_positions_strict()   # API-Fehler -> Abbruch statt Waise
    live = {(p['symbol'], _side_of(p)): p for p in positions}
    for coin, pos in list(state['open'].items()):
        key = (symbol(coin), pos['side'])
        if key not in live:
            state['closed'].append({**pos, 'coin': coin, 'closed': str(today.date()), 'reason': 'sl/tp (Boerse)'})
            del state['open'][coin]
            msgs.append(f"✅ {coin} {pos['side']} durch SL/TP geschlossen")
            continue
        held = (today - pd.Timestamp(pos['date'])).days
        if held >= MAX_DAYS:
            p = live[key]
            exchange.close_position_market(symbol(coin), pos['side'], abs(float(p['contracts'])))
            exchange.cancel_all_orders_for_symbol(symbol(coin))
            state['closed'].append({**pos, 'coin': coin, 'closed': str(today.date()), 'reason': 'zeit'})
            del state['open'][coin]
            msgs.append(f"⏱ {coin} {pos['side']} nach {held} Tagen geschlossen")
    return msgs


def open_trades(exchange, state, today, decisions, risk_pct, leverage):
    """Neue Positionen; Risiko je Trade = risk_pct des freien Guthabens bis zum Stop."""
    msgs = []
    positions = exchange.fetch_open_positions_strict()
    busy = {p['symbol'] for p in positions}
    for coin, (side, atr) in decisions.items():
        if side is None or coin in state['open'] or symbol(coin) in busy:
            continue
        balance = exchange.fetch_balance_usdt()
        ticker = exchange.fetch_ticker(symbol(coin))
        if not balance or not ticker:
            msgs.append(f"⚠️ {coin}: Guthaben/Kurs nicht abrufbar, kein Einstieg")
            continue
        entry = float(ticker['last'])
        sl, tp = levels(entry, atr, side)
        amount = balance * risk_pct / 100 / (SL_ATR * atr)
        if amount * entry > balance * leverage:
            msgs.append(f"⚠️ {coin}: Position braeuchte mehr als {leverage}x Hebel, kein Einstieg")
            continue
        exchange.set_margin_mode(symbol(coin), 'isolated')
        exchange.set_leverage(symbol(coin), leverage)
        exchange.place_market_order_with_tpsl(symbol(coin), 'buy' if side == 'long' else 'sell', amount, sl, tp)
        state['open'][coin] = {'date': str(today.date()), 'side': side, 'atr': atr, 'entry': entry,
                               'sl': sl, 'tp': tp, 'amount': amount}
        msgs.append(f"🚀 {coin} {side.upper()} {amount:.4g} @ {entry:.6g} | SL {sl:.6g} | TP {tp:.6g}")
    return msgs
