# src/wbot/live/paper.py
# Papierkonto: dieselbe Trade-Regel und Abrechnung (trade_rules.settle) wie der Backtest.
import os
import json
import pandas as pd

from wbot.forecast.trade_rules import settle

PAPER_STATE = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')),
                           'artifacts', 'live', 'paper_state.json')


def load_state(start_capital):
    if os.path.exists(PAPER_STATE):
        with open(PAPER_STATE, encoding='utf-8') as f:
            return json.load(f)
    return {'equity': float(start_capital), 'start_capital': float(start_capital), 'open': {}, 'closed': []}


def save_state(state):
    os.makedirs(os.path.dirname(PAPER_STATE), exist_ok=True)
    with open(PAPER_STATE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def step(state, today, decisions, px_by_coin, funding_by_coin, risk_pct):
    """
    Ein Tag: offene Papier-Trades abrechnen (Kerzen bis gestern), dann neue zum heutigen Open eroeffnen.
    decisions: {coin: (side|None, atr)}. Rueckgabe: (geschlossene, eroeffnete) Trades dieses Laufs.
    """
    closed, opened = [], []
    for coin, pos in list(state['open'].items()):
        px = px_by_coin[coin]
        d = pd.Timestamp(pos['date'])
        if d not in px.index:
            continue                       # Einstiegskerze noch nicht abgeschlossen
        t = px.index.get_loc(d)
        res = settle(px, funding_by_coin[coin], t, pos['side'], pos['atr'])
        if res is None:
            continue
        state['equity'] *= 1 + risk_pct / 100 * res['R']
        tr = {**pos, 'coin': coin, 'entry': float(px['open'].iloc[t]), 'exit_date': str(px.index[res['exit_idx']].date()),
              'exit_price': float(res['exit_price']), 'reason': res['reason'], 'R': round(float(res['R']), 4),
              'equity_after': round(state['equity'], 2)}
        state['closed'].append(tr)
        closed.append(tr)
        del state['open'][coin]
    for coin, (side, atr) in decisions.items():
        if side is None or coin in state['open']:
            continue
        state['open'][coin] = {'date': str(pd.Timestamp(today).date()), 'side': side, 'atr': float(atr)}
        opened.append({'coin': coin, **state['open'][coin]})
    return closed, opened
