# Schaerfe-Schritt 4: Listing-Short nach Pump-Groesse (Regeln: 04_listing_pumpgroesse.md)
import os
import json
import numpy as np
import pandas as pd

LST = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '1mbot', 'research', 'listing'))
SPLIT = pd.Timestamp('2025-07-01', tz='UTC')


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan


def pump(tag, T):
    k = pd.read_parquet(os.path.join(LST, 'ev', tag + '_k1m.parquet'))
    m0 = (int(T.value // 10**6) // 60_000) * 60_000
    pre = k.loc[k.open_time == m0 - 5 * 60_000, 'open']
    ent = k.loc[k.open_time == m0 + 30 * 60_000, 'open']
    if pre.empty or ent.empty:
        return np.nan
    return float(ent.iloc[0]) / float(pre.iloc[0]) - 1


def main():
    d = pd.read_csv(os.path.join(LST, 'diag_l2.csv'))
    d['T'] = pd.to_datetime(d['T'], utc=True)
    d['pump'] = [pump(r.tag, r.T) for r in d.itertuples()]
    d['erg'] = d['net'] + d['funding_short_pnl']
    d = d.dropna(subset=['pump', 'erg'])
    d[['tag', 'T', 'pump', 'erg', 'mae']].assign(oos=d['T'] >= SPLIT).to_csv(
        os.path.join(os.path.dirname(__file__), 's04_ereignisse.csv'), index=False)
    IS, OOS = d[d['T'] < SPLIT], d[d['T'] >= SPLIT]
    thr = IS['pump'].median()
    out = {'schwelle_pump_pct': round(thr * 100, 1), 'n_IS': len(IS), 'n_OOS': len(OOS)}
    for name, x in (('IS (nur Info)', IS), ('OOS', OOS)):
        hi, lo = x[x.pump >= thr], x[x.pump < thr]
        out[name] = {g: {'n': len(v), 'avg_pct': round(v.erg.mean() * 100, 1), 'median_pct': round(v.erg.median() * 100, 1),
                         't': round(tstat(v.erg), 2), 'hit': round((v.erg > 0).mean(), 2),
                         'mae_median_pct': round(v.mae.median() * 100, 1), 'mae_gt50_pct': round((v.mae > 0.5).mean() * 100)}
                     for g, v in (('gross', hi), ('klein', lo), ('alle', x))}
        print(name, json.dumps(out[name]))
    o = out['OOS']
    ok = o['gross']['avg_pct'] > o['klein']['avg_pct'] and o['gross']['avg_pct'] > 0 and o['gross']['t'] >= 2.0 \
        and o['gross']['mae_median_pct'] <= 1.5 * max(o['klein']['mae_median_pct'], 1e-9)
    out['bestanden'] = bool(ok)
    print('Schwelle pump:', out['schwelle_pump_pct'], '%  ==>', 'BESTANDEN' if ok else 'NICHT BESTANDEN')
    json.dump(out, open(os.path.join(os.path.dirname(__file__), 's04_ergebnis.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
