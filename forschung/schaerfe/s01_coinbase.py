# Schaerfe-Schritt 1: Coinbase-Praemie stuendlich (Regeln: 01_coinbase_stunde.md)
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from wbot.world.sources import _get, binance_funding   # noqa: E402

HDIR = os.path.join(ROOT, 'data', 'world', 'hourly')
START = pd.Timestamp('2022-01-01')
COINS = ['BTC', 'ETH']
H, TP, SL, COST = 12, 1.5, 1.0, 0.0022
VAL = ('2024-01-01', '2025-03-31')
TEST = '2025-04-01'
MARGIN = 0.05


def _cache(name, fetch):
    os.makedirs(HDIR, exist_ok=True)
    p = os.path.join(HDIR, name + '.csv')
    if os.path.exists(p) and time.time() - os.path.getmtime(p) < 6 * 3600:
        return pd.read_csv(p, index_col=0, parse_dates=True)
    df = fetch()
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df.to_csv(p)
    return df


def coinbase_1h(product):
    def fetch():
        out, cur, end = [], START, pd.Timestamp.utcnow().tz_localize(None).floor('h')
        while cur < end:
            nxt = min(cur + pd.Timedelta(hours=299), end)
            out += _get(f'https://api.exchange.coinbase.com/products/{product}/candles',
                        {'granularity': 3600, 'start': cur.strftime('%Y-%m-%dT%H:%M:%SZ'),
                         'end': nxt.strftime('%Y-%m-%dT%H:%M:%SZ')})
            cur = nxt
            time.sleep(0.2)
        return pd.DataFrame({'close': [float(r[4]) for r in out]},
                            index=pd.to_datetime([r[0] for r in out], unit='s'))
    return _cache(f'cb_{product}', fetch)


def binance_1h(sym, futures):
    url = 'https://fapi.binance.com/fapi/v1/klines' if futures else 'https://api.binance.com/api/v3/klines'
    lim = 1500 if futures else 1000

    def fetch():
        rows, cur = [], int(START.timestamp() * 1000)
        while True:
            b = _get(url, {'symbol': sym, 'interval': '1h', 'startTime': cur, 'limit': lim})
            if not b:
                break
            rows += b
            cur = b[-1][0] + 3_600_000
            if len(b) < lim:
                break
        return pd.DataFrame([[float(x) for x in r[1:5]] for r in rows], columns=['open', 'high', 'low', 'close'],
                            index=pd.to_datetime([r[0] for r in rows], unit='ms'))
    return _cache(f'bn_{"fut" if futures else "spot"}_{sym}', fetch)


def zs(x, n=168):
    return (x - x.rolling(n, min_periods=n // 2).mean()) / x.rolling(n, min_periods=n // 2).std()


def features(coin, usdt):
    cb = coinbase_1h(f'{coin}-USD')['close']
    sp = binance_1h(f'{coin}USDT', False)['close']
    prem = np.log(cb / (sp * usdt)).dropna()
    f = pd.DataFrame({'z1': zs(prem), 'z4': zs(prem.rolling(4).mean()), 'd4': zs(prem - prem.shift(4))})
    for c in list(f.columns):
        f[c + '_q'] = f[c] ** 2
    return f.shift(1)      # Stunde t sieht nur Werte bis t-1


def labels(coin):
    px = binance_1h(f'{coin}USDT', True)
    fr = binance_funding(coin)['rate']
    ft, fv = fr.index.values, np.cumsum(fr.values)
    o, h, l, c = (px[k].values for k in ('open', 'high', 'low', 'close'))
    sig = (pd.Series(np.log(c)).diff().rolling(168, min_periods=84).std() * np.sqrt(H)).shift(1).values
    idx = px.index.values
    out = []
    for t in range(len(px) - H):
        if not np.isfinite(sig[t]):
            continue
        e = o[t]; s = sig[t] * e; cR = COST * e / s
        res = {}
        for side, sg in (('L', 1), ('S', -1)):
            stop, tp = e - sg * SL * s, e + sg * TP * s
            r = None
            for j in range(t, t + H):
                if (sg == 1 and l[j] <= stop) or (sg == -1 and h[j] >= stop):
                    r, jx, win = -SL, j, 0.0; break
                if (sg == 1 and h[j] >= tp) or (sg == -1 and l[j] <= tp):
                    r, jx, win = TP, j, 1.0; break
            if r is None:
                jx = t + H - 1; r = sg * (c[jx] - e) / s; win = 0.0
            a, b = np.searchsorted(ft, idx[t], 'right'), np.searchsorted(ft, idx[jx] + np.timedelta64(1, 'h'), 'right')
            fsum = (fv[b - 1] if b else 0) - (fv[a - 1] if a else 0)
            res[side] = (r - cR - sg * fsum * e / s, win, px.index[jx])
        out.append({'ts': px.index[t], 'RL': res['L'][0], 'WL': res['L'][1], 'exL': res['L'][2],
                    'RS': res['S'][0], 'WS': res['S'][1], 'exS': res['S'][2], 'be': (1 + cR) / 2.5,
                    'resolved': px.index[t + H - 1] + pd.Timedelta(hours=1)})
    return pd.DataFrame(out).set_index('ts')


def walk_forward(panel, cols):
    res = []
    for m0 in pd.date_range('2023-01-01', panel.index.max(), freq='MS'):
        rows = panel[(panel.index >= m0) & (panel.index < m0 + pd.offsets.MonthBegin(1))].dropna(subset=cols)
        tr = panel[panel['resolved'] <= m0].dropna(subset=cols)
        if rows.empty or len(tr) < 2000:
            continue
        sc = StandardScaler().fit(tr[cols])
        X, Xr = sc.transform(tr[cols]), sc.transform(rows[cols])
        r = rows.copy()
        r['climL'], r['climS'] = tr['WL'].mean(), tr['WS'].mean()
        r['pL'] = LogisticRegression(C=0.1, max_iter=1000).fit(X, tr['WL']).predict_proba(Xr)[:, 1]
        r['pS'] = LogisticRegression(C=0.1, max_iter=1000).fit(X, tr['WS']).predict_proba(Xr)[:, 1]
        res.append(r)
    return pd.concat(res)


def evaluate(d, years):
    thr = d['be'] + MARGIN
    bss = np.mean([1 - ((d[p] - d[y]) ** 2).mean() / ((d[c] - d[y]) ** 2).mean()
                   for p, y, c in (('pL', 'WL', 'climL'), ('pS', 'WS', 'climS'))])
    sharpL, sharpS = d['pL'] >= thr, d['pS'] >= thr
    sharp = (sharpL | sharpS).mean()
    pred_hit = pd.concat([d.loc[sharpL, 'pL'], d.loc[sharpS, 'pS']])
    real_hit = pd.concat([d.loc[sharpL, 'WL'], d.loc[sharpS, 'WS']])
    be_sharp = pd.concat([d.loc[sharpL, 'be'], d.loc[sharpS, 'be']])
    trades = []
    for coin, g in d.groupby('coin'):
        free = None
        for ts, r in g.iterrows():
            if free is not None and ts < free:
                continue
            side = 'L' if (r.pL >= r.be + MARGIN and r.pL >= r.pS) else ('S' if r.pS >= r.be + MARGIN else None)
            if side is None:
                continue
            trades.append(r['R' + side])
            free = r['ex' + side] + pd.Timedelta(hours=1)
    R = np.array(trades)
    return {'bss': round(float(bss), 4), 'schaerfe_pct': round(float(sharp * 100), 2),
            'scharf_n': int(len(real_hit)), 'scharf_P_vorhergesagt': round(float(pred_hit.mean()), 3) if len(pred_hit) else None,
            'scharf_treffer_echt': round(float(real_hit.mean()), 3) if len(real_hit) else None,
            'scharf_breakeven': round(float(be_sharp.mean()), 3) if len(be_sharp) else None,
            'trades': int(len(R)), 'trades_pro_jahr': round(len(R) / years, 0),
            'avgR': round(float(R.mean()), 3) if len(R) else None,
            'pf': round(float(R[R > 0].sum() / max(-R[R < 0].sum(), 1e-9)), 2) if len(R) else None,
            'klima_L': round(float(d['climL'].mean()), 3), 'p_q99': round(float(np.nanquantile(d[['pL', 'pS']].values, 0.99)), 3)}


def main():
    usdt = coinbase_1h('USDT-USD')['close']
    parts = []
    for coin in COINS:
        print(f'{coin}: Daten + Ergebnisse ...', flush=True)
        f = features(coin, usdt)
        lab = labels(coin)
        parts.append(lab.join(f, how='left').assign(coin=coin))
    panel = pd.concat(parts).sort_index()
    cols = [c for c in panel.columns if c[:2] in ('z1', 'z4', 'd4')]
    wf = walk_forward(panel, cols)
    out = {}
    for name, d, yrs in (('validierung', wf[(wf.index >= VAL[0]) & (wf.index <= VAL[1] + ' 23:59')], 1.25),
                         ('endtest', wf[wf.index >= TEST], (wf.index.max() - pd.Timestamp(TEST)).days / 365)):
        out[name] = evaluate(d, yrs)
        print(name, json.dumps(out[name]))
    ok = all(v['schaerfe_pct'] >= 2 and v['scharf_treffer_echt'] is not None
             and v['scharf_treffer_echt'] >= v['scharf_breakeven'] + 0.02 and v['trades'] >= 50
             and (v['avgR'] or -1) > 0 and v['bss'] > 0 for v in out.values())
    out['bestanden'] = ok
    print('==>', 'BESTANDEN' if ok else 'NICHT BESTANDEN')
    json.dump(out, open(os.path.join(os.path.dirname(__file__), 's01_ergebnis.json'), 'w'), indent=2)
    wf.to_csv(os.path.join(HDIR, 's01_walkforward.csv'))


if __name__ == '__main__':
    main()
