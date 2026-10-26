# tests/test_wbot.py -- ohne Netzwerk (synthetische Daten)
import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from wbot.forecast.trade_rules import settle, label_table, levels, MAX_DAYS, TP_ATR, SL_ATR, COST_PCT
from wbot.forecast.ensemble import combine, decide
from wbot.world.features import _align
from wbot.live import paper


def _px(n=120, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    o = np.r_[100, c[:-1]]
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.02, n))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.02, n))
    idx = pd.date_range('2024-01-01', periods=n, freq='D')
    return pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c}, index=idx)


def _flat(n=30):
    idx = pd.date_range('2024-01-01', periods=n, freq='D')
    return pd.DataFrame({'open': 100.0, 'high': 100.5, 'low': 99.5, 'close': 100.0}, index=idx)


def test_tp_hit_long():
    px = _flat()
    px.loc[px.index[3], 'high'] = 103.0
    r = settle(px, None, 1, 'long', 2.0)
    assert r['reason'] == 'tp' and r['exit_idx'] == 3
    assert r['R'] == pytest.approx(TP_ATR - COST_PCT * 100 / 2.0)


def test_sl_wins_same_candle():
    px = _flat()
    px.loc[px.index[2], ['high', 'low']] = [104.0, 97.0]
    r = settle(px, None, 1, 'long', 2.0)
    assert r['reason'] == 'sl' and r['R'] == pytest.approx(-SL_ATR - COST_PCT * 100 / 2.0)


def test_time_exit_and_open():
    px = _flat()
    r = settle(px, None, 0, 'short', 2.0)
    assert r['reason'] == 'zeit' and r['exit_idx'] == MAX_DAYS - 1
    assert settle(px, None, 0, 'short', 2.0, upto=MAX_DAYS - 2) is None


def test_funding_cost_sign():
    px = _flat()
    fund = pd.Series(0.001, index=pd.date_range('2024-01-01 08:00', periods=60, freq='8h'))
    rl = settle(px, fund, 0, 'long', 2.0)['R']
    rs = settle(px, fund, 0, 'short', 2.0)['R']
    assert rl < rs          # Long zahlt positives Funding, Short bekommt es


def test_levels():
    assert levels(100, 2, 'long') == (98, 103)
    assert levels(100, 2, 'short') == (102, 97)


def test_align_no_lookahead():
    """Ein Wert vom Datentag D darf fruehestens am Entscheidungstag D+1+extra_lag erscheinen."""
    idx = pd.date_range('2024-01-01', periods=10, freq='D')
    df = pd.DataFrame({'x': np.arange(10.0)}, index=idx)
    days = pd.date_range('2024-01-01', periods=11, freq='D')
    a0 = _align(df, days, 0)['x']
    a2 = _align(df, days, 2)['x']
    assert a0.loc['2024-01-05'] == 3.0        # Datentag 04.01.
    assert a2.loc['2024-01-05'] == 1.0        # Datentag 02.01.
    df2 = df.copy()
    df2.loc['2024-01-05':, 'x'] = 999         # Zukunft aendern -> Vergangenheit unveraendert
    assert _align(df2, days, 0)['x'].loc[:'2024-01-05'].equals(a0.loc[:'2024-01-05'])


def test_combine_and_decide():
    clim = (0.35, 0.32)
    pL, pS = combine({}, clim)
    assert len(pL) == 0
    pL, pS = combine({'a': (np.array([0.35, np.nan]), np.array([0.32, 0.32]))}, clim)
    assert pL == pytest.approx([0.35, 0.35]) and pS == pytest.approx([0.32, 0.32])
    pL, _ = combine({'a': (np.array([0.45]), np.array([0.32])), 'b': (np.array([0.45]), np.array([0.32]))}, clim)
    assert pL[0] > 0.45                       # zwei unabhaengige Stationen verstaerken sich
    sides = decide(np.array([0.5, 0.3, 0.46]), np.array([0.3, 0.5, 0.47]))
    assert list(sides) == ['long', 'short', 'short']


def test_paper_equals_backtest():
    """Papierkonto Tag fuer Tag == Abrechnung im Backtest (label_table) bei gleichen Entscheidungen."""
    px = _px(160)
    lab = label_table(px, None)
    rng = np.random.default_rng(1)
    want = {d: rng.choice(['long', 'short', None]) for d in lab.index}
    atr = (pd.concat([px['high'] - px['low'], (px['high'] - px['close'].shift()).abs(),
                      (px['low'] - px['close'].shift()).abs()], axis=1).max(axis=1)).rolling(14).mean()
    # Backtest: eine Position, frei ab Ausstiegstag + 1
    bt, free = [], None
    for d, r in lab.iterrows():
        if (free is not None and d < free) or want[d] is None:
            continue
        bt.append(round(r.RL if want[d] == 'long' else r.RS, 4))
        free = (r.exitL if want[d] == 'long' else r.exitS) + pd.Timedelta(days=1)
    # Papier: taeglich nur Kerzen bis gestern sichtbar
    st = {'equity': 1000.0, 'start_capital': 1000.0, 'open': {}, 'closed': []}
    for d in lab.index:
        t = px.index.get_loc(d)
        seen = px.iloc[:t]
        paper.step(st, d, {'X': (want[d], float(atr.iloc[t - 1]))}, {'X': seen}, {'X': None}, 1.0)
    pp = [round(c['R'], 4) for c in st['closed']]
    assert pp == bt[:len(pp)] and len(bt) - len(pp) <= 1
