# src/wbot/analysis/research.py
# Praeregistrierter Test (forschung/PRAEREGISTRIERUNG.md): Stationsauswahl in der Validierung,
# Ensemble-Pruefung im unberuehrten Endtest. Schreibt artifacts/results/research.json + walkforward.csv.
import os
import sys
import json
import logging
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from wbot.world.features import STATIONS
from wbot.forecast.ensemble import build_panel, walk_forward, combine, decide
from wbot.forecast.trade_rules import P_MIN

PRED_START = '2023-01-01'
VAL = ('2024-01-01', '2025-03-31')
TEST_START = '2025-04-01'
RESULTS = os.path.join(PROJECT_ROOT, 'artifacts', 'results')


def bss(p, y, clim):
    m = np.isfinite(p) & np.isfinite(y)
    if m.sum() < 30:
        return np.nan
    return 1 - np.mean((p[m] - y[m]) ** 2) / np.mean((clim[m] - y[m]) ** 2)


def simulate(df, pL, pS):
    """Eine Position je Coin; liefert Liste von Trades (coin, Datum, Seite, R, Ausstieg)."""
    df = df.assign(_pL=pL, _pS=pS, side_=decide(pL, pS))
    trades = []
    for coin, g in df.groupby('coin'):
        free_from = None
        for ts, r in g.iterrows():
            if free_from is not None and ts < free_from:
                continue
            if r.side_ is None or not np.isfinite(r.RL):
                continue
            long_ = r.side_ == 'long'
            exit_ = r.exitL if long_ else r.exitS
            trades.append({'coin': coin, 'date': ts, 'side': r.side_, 'R': r.RL if long_ else r.RS, 'exit': exit_})
            free_from = exit_ + pd.Timedelta(days=1)
    return pd.DataFrame(trades)


def stats(trades, df, pL, pS, years, risk_pct=1.0):
    y = df
    b = np.nanmean([bss(pL, y['WL'].values, y['climL'].values), bss(pS, y['WS'].values, y['climS'].values)])
    if trades.empty:
        return {'bss': b, 'trades': 0, 'trades_per_year': 0, 'avgR': np.nan, 'pf': np.nan, 'coins_pos': 0}
    R = trades['R'].values
    pf = R[R > 0].sum() / max(-R[R < 0].sum(), 1e-9)
    per_coin = trades.groupby('coin')['R'].sum()
    eq = np.cumprod(1 + risk_pct / 100 * trades.sort_values('exit')['R'].values)
    dd = float(np.max(1 - eq / np.maximum.accumulate(eq)) * 100)
    return {'bss': round(float(b), 4), 'trades': int(len(R)), 'trades_per_year': round(len(R) / years, 1),
            'avgR': round(float(R.mean()), 3), 'pf': round(float(pf), 3),
            'coins_pos': int((per_coin > 0).sum()), 'per_coin_R': per_coin.round(1).to_dict(),
            'longs': int((trades.side == 'long').sum()), 'shorts': int((trades.side == 'short').sum()),
            f'rendite_pa_{risk_pct:g}pct_risk': round(float((eq[-1] ** (1 / years) - 1) * 100), 1),
            'max_dd_pct': round(dd, 1)}


def main():
    logging.basicConfig(level=logging.WARNING)
    os.makedirs(RESULTS, exist_ok=True)
    print('Baue Datensatz (Merkmale + Ergebnisse) ...', flush=True)
    panel = build_panel()
    end = panel.dropna(subset=['RL']).index.max()
    print(f'Walk-Forward {PRED_START} .. {end.date()} ...', flush=True)
    wf = walk_forward(panel, STATIONS, PRED_START, end)
    wf = wf.dropna(subset=['RL', 'RS'])
    wf.to_csv(os.path.join(RESULTS, 'walkforward.csv'))

    val = wf[(wf.index >= VAL[0]) & (wf.index <= VAL[1])]
    test = wf[wf.index >= TEST_START]
    yv = (pd.Timestamp(VAL[1]) - pd.Timestamp(VAL[0])).days / 365
    yt = (test.index.max() - pd.Timestamp(TEST_START)).days / 365

    out = {'validierung': {}, 'test_info_je_station': {}}
    accepted = []
    print(f'\n=== VALIDIERUNG {VAL[0]} .. {VAL[1]} (Stationsauswahl) ===')
    for st in STATIONS:
        pL, pS = val[f'{st}_pL'].values, val[f'{st}_pS'].values
        bL = bss(pL, val['WL'].values, val['climL'].values)
        bS = bss(pS, val['WS'].values, val['climS'].values)
        s = stats(simulate(val, pL, pS), val, pL, pS, yv)
        ok = (np.nan_to_num(bL, nan=-1) > 0 or np.nan_to_num(bS, nan=-1) > 0) and s['trades'] >= 30 and s['avgR'] > 0
        if ok:
            accepted.append(st)
        out['validierung'][st] = {'bss_long': round(float(bL), 4), 'bss_short': round(float(bS), 4), **s,
                                  'aufgenommen': bool(ok)}
        print(f"  {st:15s} BSS L={bL:+.4f} S={bS:+.4f} | Trades {s['trades']:4d} avgR={s['avgR']:+.3f} "
              f"PF={s['pf']:.2f} -> {'AUFGENOMMEN' if ok else 'raus'}")

    print(f'\nAufgenommene Stationen: {accepted or "KEINE"}')
    out['aufgenommen'] = accepted

    print(f'\n=== ENDTEST {TEST_START} .. {test.index.max().date()} ({yt:.2f} J) ===')
    if accepted:
        preds = {st: (test[f'{st}_pL'].values, test[f'{st}_pS'].values) for st in accepted}
        pL, pS = combine(preds, (test['climL'].values, test['climS'].values))
        s = stats(simulate(test, pL, pS), test, pL, pS, yt)
        passed = (s['bss'] > 0 and s['pf'] >= 1.2 and s['avgR'] >= 0.1 and s['coins_pos'] >= 4
                  and s['trades_per_year'] >= 100)
    else:
        s, passed = {}, False
    out['endtest_ensemble'] = s
    out['bestanden'] = bool(passed)
    print('  Ensemble:', json.dumps(s, default=str))
    print(f"  ==> {'BESTANDEN' if passed else 'NICHT BESTANDEN'}")

    print('\n(nur Info, nicht entscheidungsrelevant) Endtest je Station:')
    for st in STATIONS:
        pL, pS = test[f'{st}_pL'].values, test[f'{st}_pS'].values
        s2 = stats(simulate(test, pL, pS), test, pL, pS, yt)
        out['test_info_je_station'][st] = s2
        print(f"  {st:15s} BSS={s2['bss']:+.4f} Trades {s2['trades']:4d} avgR={s2['avgR']:+.3f} PF={s2['pf']:.2f}")
    out['erstellt'] = pd.Timestamp.utcnow().isoformat()
    json.dump(out, open(os.path.join(RESULTS, 'research.json'), 'w', encoding='utf-8'), indent=2, default=str)


if __name__ == '__main__':
    main()
