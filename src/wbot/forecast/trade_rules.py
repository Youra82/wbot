# src/wbot/forecast/trade_rules.py
# Trade-Regel (TP/SL/Haltedauer) und Abrechnung eines Trades auf Tageskerzen.
# Gemeinsam fuer Forschung, Backtest und Papiermodus -> Papier == Backtest.
import numpy as np
import pandas as pd

TP_ATR = 1.5
SL_ATR = 1.0
MAX_DAYS = 10
COST_PCT = 0.0022          # Gebuehren + Schlupf je Runde (Taker 2x0,06 % + 2x0,05 %)
P_MIN = 0.45               # Mindest-Wahrscheinlichkeit fuer einen Trade


def levels(entry, atr, side):
    if side == 'long':
        return entry - SL_ATR * atr, entry + TP_ATR * atr
    return entry + SL_ATR * atr, entry - TP_ATR * atr


def settle(px, funding, t, side, atr, entry=None, upto=None):
    """
    Rechnet einen Trade ab, der zum Open von Tag t (Index in px) eroeffnet wird.
    px: Tageskerzen (Index = Datum), funding: Serie Funding-Raten (Index = Zeitstempel).
    upto: letzter verfuegbarer Kerzenindex (Live: gestern); None = alles.
    Rueckgabe dict(R, exit_idx, reason, exit_price) oder None, wenn noch offen.
    """
    o, h, l, c = (px[k].values for k in ('open', 'high', 'low', 'close'))
    entry = o[t] if entry is None else entry
    sl, tp = levels(entry, atr, side)
    last = len(px) - 1 if upto is None else upto
    end = t + MAX_DAYS - 1
    exit_idx = exit_px = reason = None
    for j in range(t, min(end, last) + 1):
        if (side == 'long' and l[j] <= sl) or (side == 'short' and h[j] >= sl):
            exit_idx, exit_px, reason = j, sl, 'sl'
            break
        if (side == 'long' and h[j] >= tp) or (side == 'short' and l[j] <= tp):
            exit_idx, exit_px, reason = j, tp, 'tp'
            break
    if exit_idx is None:
        if end > last:
            return None
        exit_idx, exit_px, reason = end, c[end], 'zeit'
    sign = 1 if side == 'long' else -1
    gross = sign * (exit_px - entry) / atr
    t0 = px.index[t] + pd.Timedelta(minutes=1)
    t1 = px.index[exit_idx] + pd.Timedelta(days=1)
    fsum = float(funding[(funding.index > t0) & (funding.index <= t1)].sum()) if funding is not None else 0.0
    cost = (COST_PCT + sign * fsum) * entry / atr
    return {'R': gross - cost, 'exit_idx': exit_idx, 'reason': reason, 'exit_price': exit_px, 'tp_hit': reason == 'tp'}


def label_table(px, funding):
    """Ergebnis von Long und Short fuer jeden Entscheidungstag (Forschung/Training)."""
    atr = (pd.concat([px['high'] - px['low'], (px['high'] - px['close'].shift()).abs(),
                      (px['low'] - px['close'].shift()).abs()], axis=1).max(axis=1)).rolling(14).mean()
    atr_prev = atr.shift(1).values       # ATR aus Kerzen <= t-1
    rows = []
    for t in range(len(px)):
        if not np.isfinite(atr_prev[t]):
            continue
        L = settle(px, funding, t, 'long', atr_prev[t])
        S = settle(px, funding, t, 'short', atr_prev[t])
        if L is None or S is None:
            continue
        rows.append({'date': px.index[t], 'RL': L['R'], 'RS': S['R'], 'WL': float(L['tp_hit']),
                     'WS': float(S['tp_hit']), 'exitL': px.index[L['exit_idx']], 'exitS': px.index[S['exit_idx']],
                     'resolved': px.index[max(L['exit_idx'], S['exit_idx'])] + pd.Timedelta(days=1)})
    return pd.DataFrame(rows).set_index('date')
