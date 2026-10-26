# Schaerfe-Schritt 2: Korea-Praemie stuendlich (Regeln: 02_korea_stunde.md)
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from s01_coinbase import _cache, binance_1h, zs, START, VAL, TEST, HDIR, _get  # noqa: E402
from s01b_coinbase import labels                                                # noqa: E402

COINS = ['BTC', 'ETH', 'XRP', 'SOL', 'DOGE']
Q = 0.90


def upbit_1h(coin):
    def fetch():
        out, to = [], None
        while True:
            p = {'market': f'KRW-{coin}', 'count': 200}
            if to:
                p['to'] = to
            b = _get('https://api.upbit.com/v1/candles/minutes/60', p)
            if not b:
                break
            out += b
            first = pd.Timestamp(b[-1]['candle_date_time_utc'])
            if first <= START or len(b) < 200:
                break
            to = first.strftime('%Y-%m-%dT%H:%M:%SZ')
            time.sleep(0.12)
        return pd.DataFrame({'close': [float(r['trade_price']) for r in out]},
                            index=pd.to_datetime([r['candle_date_time_utc'] for r in out]))
    return _cache(f'upbit_{coin}', fetch)


def premium_dev(coin):
    kp = np.log(upbit_1h(coin)['close'] / binance_1h(f'{coin}USDT', False)['close']).dropna()
    return kp - kp.rolling(168, min_periods=84).mean()


def features(coin, btc_dev):
    dev = premium_dev(coin)
    f = pd.DataFrame({'k1': zs(dev), 'k4': zs(dev.rolling(4).mean()), 'kd4': zs(dev - dev.shift(4)),
                      'kbtc1': zs(btc_dev)})
    for c in list(f.columns):
        f[c + '_q'] = f[c] ** 2
    return f.shift(1)


def walk_forward(panel, cols):
    res = []
    for m0 in pd.date_range('2023-01-01', panel.index.max(), freq='MS'):
        rows = panel[(panel.index >= m0) & (panel.index < m0 + pd.offsets.MonthBegin(1))].dropna(subset=cols)
        tr = panel[panel['resolved'] <= m0].dropna(subset=cols)
        if rows.empty or len(tr) < 5000:
            continue
        sc = StandardScaler().fit(tr[cols])
        X, Xr = sc.transform(tr[cols]), sc.transform(rows[cols])
        r = rows.copy()
        r['climL'], r['climS'] = tr['WL'].mean(), tr['WS'].mean()
        for s in ('L', 'S'):
            m = LogisticRegression(C=0.1, max_iter=1000).fit(X, tr['W' + s])
            r['p' + s] = m.predict_proba(Xr)[:, 1]
            r['q' + s] = np.quantile(m.predict_proba(X)[:, 1], Q)     # Schwelle nur aus Training
        res.append(r)
    return pd.concat(res)


def evaluate(d, years):
    bss = np.mean([1 - ((d[p] - d[y]) ** 2).mean() / ((d[c] - d[y]) ** 2).mean()
                   for p, y, c in (('pL', 'WL', 'climL'), ('pS', 'WS', 'climS'))])
    sigL, sigS = d['pL'] > d['qL'], d['pS'] > d['qS']
    sig_R = pd.concat([d.loc[sigL, 'RML'], d.loc[sigS, 'RMS']]).dropna()
    sig_T = pd.concat([d.loc[sigL, 'RTL'], d.loc[sigS, 'RTS']])
    trades = []
    for coin, g in d.groupby('coin'):
        free = None
        for ts, r in g.iterrows():
            if free is not None and ts < free:
                continue
            okL, okS = r.pL > r.qL, r.pS > r.qS
            if okL and okS:
                side = 'L' if r.pL / r.qL >= r.pS / r.qS else 'S'
            else:
                side = 'L' if okL else ('S' if okS else None)
            if side is None or not np.isfinite(r['RM' + side]):
                continue
            trades.append(r['RM' + side])
            free = r['exM' + side] + pd.Timedelta(hours=1)
    R = np.array(trades)
    return {'bss': round(float(bss), 4), 'signal_anteil_pct': round(float((sigL | sigS).mean() * 100), 1),
            'signal_n_ausgefuehrt': int(len(sig_R)), 'signal_avgR_maker': round(float(sig_R.mean()), 3),
            'signal_avgR_taker': round(float(sig_T.mean()), 3),
            'alle_avgR_maker': round(float(pd.concat([d['RML'], d['RMS']]).dropna().mean()), 3),
            'trades': int(len(R)), 'trades_pro_jahr': round(len(R) / years),
            'avgR': round(float(R.mean()), 3) if len(R) else None,
            'pf': round(float(R[R > 0].sum() / max(-R[R < 0].sum(), 1e-9)), 2) if len(R) else None,
            'summeR': round(float(R.sum()), 1)}


def main():
    btc_dev = premium_dev('BTC')
    parts = []
    for c in COINS:
        print(f'{c} ...', flush=True)
        parts.append(labels(c).join(features(c, btc_dev), how='left').assign(coin=c))
    panel = pd.concat(parts).sort_index()
    cols = [c for c in panel.columns if c[:1] == 'k']
    wf = walk_forward(panel, cols)
    wf.to_csv(os.path.join(HDIR, 's02_walkforward.csv'))
    out = {}
    for name, d, yrs in (('validierung', wf[(wf.index >= VAL[0]) & (wf.index <= VAL[1] + ' 23:59')], 1.25),
                         ('endtest', wf[wf.index >= TEST], (wf.index.max() - pd.Timestamp(TEST)).days / 365)):
        out[name] = evaluate(d, yrs)
        print(name, json.dumps(out[name]))
    ok = all(v['signal_avgR_maker'] > 0 and v['trades'] >= 50 and (v['avgR'] or -1) > 0 and v['bss'] > 0
             for v in out.values())
    out['bestanden'] = bool(ok)
    print('==>', 'BESTANDEN' if ok else 'NICHT BESTANDEN')
    json.dump(out, open(os.path.join(os.path.dirname(__file__), 's02_ergebnis.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
