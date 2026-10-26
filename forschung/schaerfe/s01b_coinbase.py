# Schaerfe-Schritt 1b: symmetrisches Ziel + Maker-Einstieg (Regeln: 01b_coinbase_symmetrisch_maker.md)
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from s01_coinbase import (coinbase_1h, binance_1h, binance_funding, features, walk_forward,  # noqa: E402
                          COINS, H, VAL, TEST, HDIR)

TPSL = 1.0
OFF = 0.1                      # Limit 0,1 sigma unter/ueber dem Open
MAKER, TAKER, SLIP = 0.0002, 0.0006, 0.0005
COST_T = 0.0022
MARGIN = 0.03


def labels(coin):
    px = binance_1h(f'{coin}USDT', True)
    fr = binance_funding(coin)['rate']
    ft, fv = fr.index.values, np.cumsum(fr.values)
    o, h, l, c = (px[k].values for k in ('open', 'high', 'low', 'close'))
    sig = (pd.Series(np.log(c)).diff().rolling(168, min_periods=84).std() * np.sqrt(H)).shift(1).values
    idx = px.index.values

    def fund(t0, t1):
        a, b = np.searchsorted(ft, idx[t0], 'right'), np.searchsorted(ft, idx[t1] + np.timedelta64(1, 'h'), 'right')
        return (fv[b - 1] if b else 0) - (fv[a - 1] if a else 0)

    def run(e, s, sg, t, skip_tp_first):
        stop, tp = e - sg * TPSL * s, e + sg * TPSL * s
        for j in range(t, t + H):
            if (sg == 1 and l[j] <= stop) or (sg == -1 and h[j] >= stop):
                return -TPSL, j, 'sl'
            if j == t and skip_tp_first:
                continue
            if (sg == 1 and h[j] >= tp) or (sg == -1 and l[j] <= tp):
                return TPSL, j, 'tp'
        j = t + H - 1
        return sg * (c[j] - e) / s, j, 'zeit'

    rows = []
    for t in range(len(px) - H):
        if not np.isfinite(sig[t]):
            continue
        s = sig[t] * o[t]
        row = {'ts': px.index[t], 'resolved': px.index[t + H - 1] + pd.Timedelta(hours=1)}
        for side, sg in (('L', 1), ('S', -1)):
            # Label + Variante T: Market zum Open
            r, j, why = run(o[t], s, sg, t, False)
            cT = COST_T * o[t] / s
            row[f'W{side}'] = float(why == 'tp')
            row[f'RT{side}'] = r - cT - sg * fund(t, j) * o[t] / s
            row[f'exT{side}'] = px.index[j]
            row[f'beT{side}'] = (1 + cT) / 2
            # Variante M: Limit, nur bei echtem Durchschreiten ausgefuehrt
            lim = o[t] - sg * OFF * s
            filled = (l[t] < lim) if sg == 1 else (h[t] > lim)
            cM_tp = (MAKER + MAKER) * lim / s
            cM_sl = (MAKER + TAKER + SLIP) * lim / s
            row[f'beM{side}'] = (1 + (cM_tp + cM_sl) / 2) / 2      # (1 + c) / 2 mit mittleren Kosten
            if filled:
                r, j, why = run(lim, s, sg, t, True)
                cost = cM_tp if why == 'tp' else cM_sl
                row[f'RM{side}'] = r - cost - sg * fund(t, j) * lim / s
                row[f'exM{side}'] = px.index[j]
            else:
                row[f'RM{side}'] = np.nan
                row[f'exM{side}'] = pd.NaT
        rows.append(row)
    return pd.DataFrame(rows).set_index('ts')


def evaluate(d, years, var):
    bss = np.mean([1 - ((d[p] - d[y]) ** 2).mean() / ((d[c] - d[y]) ** 2).mean()
                   for p, y, c in (('pL', 'WL', 'climL'), ('pS', 'WS', 'climS'))])
    sharp = ((d['pL'] >= d[f'be{var}L'] + MARGIN) | (d['pS'] >= d[f'be{var}S'] + MARGIN)).mean()
    trades = []
    for coin, g in d.groupby('coin'):
        free = None
        for ts, r in g.iterrows():
            if free is not None and ts < free:
                continue
            okL, okS = r.pL >= r[f'be{var}L'] + MARGIN, r.pS >= r[f'be{var}S'] + MARGIN
            side = 'L' if okL and (r.pL >= r.pS or not okS) else ('S' if okS else None)
            if side is None or not np.isfinite(r[f'R{var}{side}']):
                continue              # kein Signal oder Limit nicht ausgefuehrt
            trades.append((r[f'R{var}{side}'], r[f'W{side}'], r[f'p{side}'], r[f'be{var}{side}']))
            free = r[f'ex{var}{side}'] + pd.Timedelta(hours=1)
    if not trades:
        return {'bss': round(float(bss), 4), 'schaerfe_pct': round(float(sharp * 100), 2), 'trades': 0}
    R, W, P, BE = map(np.array, zip(*trades))
    return {'bss': round(float(bss), 4), 'schaerfe_pct': round(float(sharp * 100), 2),
            'trades': int(len(R)), 'trades_pro_jahr': round(len(R) / years),
            'P_vorhergesagt': round(float(P.mean()), 3), 'treffer_echt': round(float((R > 0).mean()), 3),
            'tp_label_treffer': round(float(W.mean()), 3), 'breakeven': round(float(BE.mean()), 3),
            'avgR': round(float(R.mean()), 3), 'pf': round(float(R[R > 0].sum() / max(-R[R < 0].sum(), 1e-9)), 2),
            'summeR': round(float(R.sum()), 1), 'klima_L': round(float(d['climL'].mean()), 3)}


def main():
    usdt = coinbase_1h('USDT-USD')['close']
    parts = [labels(c).join(features(c, usdt), how='left').assign(coin=c) for c in COINS]
    panel = pd.concat(parts).sort_index()
    cols = [c for c in panel.columns if c[:2] in ('z1', 'z4', 'd4')]
    wf = walk_forward(panel, cols)
    out = {}
    for name, d, yrs in (('validierung', wf[(wf.index >= VAL[0]) & (wf.index <= VAL[1] + ' 23:59')], 1.25),
                         ('endtest', wf[wf.index >= TEST], (wf.index.max() - pd.Timestamp(TEST)).days / 365)):
        for var in ('M', 'T'):
            out[f'{name}_{var}'] = evaluate(d, yrs, var)
            print(f'{name} {var}:', json.dumps(out[f"{name}_{var}"]))
    ok = all(v['schaerfe_pct'] >= 2 and v['trades'] >= 50 and v['avgR'] > 0 and v['bss'] > 0
             and v['treffer_echt'] >= v['breakeven'] + 0.02
             for k, v in out.items() if k.endswith('_M') and v['trades'] > 0) and \
        all(out[k]['trades'] > 0 for k in ('validierung_M', 'endtest_M'))
    out['bestanden'] = bool(ok)
    print('==>', 'BESTANDEN' if ok else 'NICHT BESTANDEN')
    json.dump(out, open(os.path.join(os.path.dirname(__file__), 's01b_ergebnis.json'), 'w'), indent=2)
    wf.to_csv(os.path.join(HDIR, 's01b_walkforward.csv'))


if __name__ == '__main__':
    main()
