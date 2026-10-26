# Schaerfe-Schritt 3: Wal-Einzahlungen schaerfen (Regeln: 03_wal_einzahlungen.md)
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

KBOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'kbot'))
sys.path.insert(0, os.path.join(KBOT, 'src'))
sys.path.insert(0, os.path.join(KBOT, 'research', 'cex'))
from test import load, EV_START, SPLIT                    # noqa: E402  (kbot-Auswertung)
from collect import D, TOKENS, load_px                    # noqa: E402
from kbot.strategy.deposit_signal import DepositRule, entry_minute  # noqa: E402

A, H, COST = 2.5e5, 60, 0.0012
FEATS = ['log_usd', 'vor60', 'wal', 'cluster', 'btc60']


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan


def events():
    btc = load_px('BTC')
    btc = btc.reindex(pd.date_range(btc.index[0], btc.index[-1], freq='min', tz='UTC')).ffill()
    btc = btc.iloc[:, 0] if isinstance(btc, pd.DataFrame) else btc
    rows = []
    for tok in TOKENS:
        if not os.path.exists(os.path.join(D, 'deposits', f'{tok}.json')):
            continue
        px, sw, dp = load(tok)
        px = px.iloc[:, 0] if isinstance(px, pd.DataFrame) else px
        if not len(dp):
            continue
        big = dp[(dp.usd >= A)].sort_values('ts')
        ev = big[big.ts >= EV_START]
        rule = DepositRule(A, H)
        for _, e in ev.iterrows():
            if not rule.accept(tok, e.ts, e.usd, e.known_since):
                continue
            t0 = entry_minute(e.ts)
            t_pre, t_end = t0 - pd.Timedelta(minutes=60), t0 + pd.Timedelta(minutes=H)
            if t_pre not in px.index or t_end not in px.index or t0 not in btc.index or t_pre not in btc.index:
                continue
            p0 = px[t0]
            rows.append({'tok': tok, 't': t0, 'usd': e.usd,
                         'log_usd': np.log(e.usd), 'vor60': p0 / px[t_pre] - 1, 'wal': float(e.cex == '-'),
                         'cluster': int(((big.ts < e.ts) & (big.ts >= e.ts - pd.Timedelta(minutes=60))).sum()),
                         'btc60': btc[t0] / btc[t_pre] - 1,
                         'short': -(px[t_end] / p0 - 1)})
    return pd.DataFrame(rows).dropna()


def main():
    ev = events()
    ev['oos'] = ev.t >= SPLIT
    IS, OOS = ev[~ev.oos], ev[ev.oos].copy()
    sc = StandardScaler().fit(IS[FEATS])
    m = Ridge(alpha=1.0).fit(sc.transform(IS[FEATS]), IS['short'])
    OOS['pred'] = m.predict(sc.transform(OOS[FEATS]))
    IS = IS.assign(pred=m.predict(sc.transform(IS[FEATS])))
    out = {'koeffizienten': dict(zip(FEATS, np.round(m.coef_ * 1e4, 2))), 'einheit_koeff': 'Basispunkte je Std.-Abw.'}
    for name, d in (('IS (angepasst, nur Info)', IS), ('OOS', OOS)):
        top = d[d.pred >= d.pred.quantile(0.7)]
        net_all, net_top = d.short - COST, top.short - COST
        q = pd.qcut(d.pred, 5, labels=False)
        out[name] = {'n': len(d), 'alle_netto_pct': round(net_all.mean() * 100, 3), 't_alle': round(tstat(net_all), 2),
                     'top30_n': len(top), 'top30_netto_pct': round(net_top.mean() * 100, 3),
                     't_top30': round(tstat(net_top), 2),
                     'spearman': round(spearmanr(d.pred, d.short).correlation, 3),
                     'quintile_brutto_pct': (d.groupby(q).short.mean() * 100).round(3).tolist()}
        print(name, json.dumps(out[name]))
    o = out['OOS']
    ok = o['top30_netto_pct'] > 0 and o['t_top30'] >= 1.5 and o['top30_netto_pct'] > o['alle_netto_pct'] \
        and o['spearman'] > 0
    out['bestanden'] = bool(ok)
    print('Koeffizienten (bp/Std):', out['koeffizienten'])
    print('==>', 'BESTANDEN' if ok else 'NICHT BESTANDEN')
    json.dump(out, open(os.path.join(os.path.dirname(__file__), 's03_ergebnis.json'), 'w'), indent=2, default=float)


if __name__ == '__main__':
    main()
