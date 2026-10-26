# Schaerfe-Schritt 5: Funding-Querschnitt marktneutral (Regeln: 05_funding_querschnitt.md)
import os
import sys
import json
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from wbot.world.sources import binance_klines, binance_funding   # noqa: E402

UNIV = ('BTC ETH BNB SOL XRP DOGE ADA AVAX LINK DOT LTC BCH TRX ETC ATOM NEAR FIL APT ARB OP INJ SUI AAVE '
        'UNI XLM HBAR ICP SAND MANA AXS CRV LDO RUNE GALA CHZ DYDX FET GRT EGLD THETA').split()
COST = 0.0011
PERIODS = {'lernen (Info)': ('2021-01-01', '2023-12-31'), 'validierung': ('2024-01-01', '2025-03-31'),
           'endtest': ('2025-04-01', '2100-01-01')}


def main():
    ret, sig, fday = {}, {}, {}
    for c in UNIV:
        try:
            px, fr = binance_klines(c), binance_funding(c)['rate']
        except Exception as e:
            print(f'{c}: uebersprungen ({e})')
            continue
        o = px['open']
        ret[c] = o.shift(-1) / o - 1                               # Open t -> Open t+1
        # Haltetag t = 00:05 t bis 00:05 t+1 -> Zahlungen 08:00 t, 16:00 t, 00:00 t+1
        day = (fr.index - pd.Timedelta(minutes=1)).normalize()
        daily = fr.groupby(day).sum()
        fday[c] = daily
        cnt = fr.groupby(day).size()
        mean3 = daily.rolling(3).sum() / cnt.rolling(3).sum()
        sig[c] = mean3.shift(1)                                    # bis Tag t-1 bekannt -> Entscheidung t
    R = pd.DataFrame(ret)
    S = pd.DataFrame(sig).reindex(R.index)
    F = pd.DataFrame(fday).reindex(R.index).fillna(0)
    hist = R.notna().cumsum()
    S = S.where(hist >= 60)
    w_prev = pd.Series(0.0, index=R.columns)
    rows = []
    for t in R.index:
        s = S.loc[t].dropna()
        r = R.loc[t]
        if len(s) < 10 or r[s.index].isna().any():
            continue
        k = max(1, len(s) // 5)
        order = s.sort_values()
        w = pd.Series(0.0, index=R.columns)
        w[order.index[:k]] = 0.5 / k                               # Long niedrigstes Funding
        w[order.index[-k:]] = -0.5 / k                             # Short hoechstes Funding
        pnl = (w * r.fillna(0)).sum() - (w * F.loc[t]).sum()       # Long zahlt +Rate, Short erhaelt
        cost = (w - w_prev).abs().sum() * COST
        rows.append({'t': t, 'r': pnl - cost, 'turnover': (w - w_prev).abs().sum(), 'btc': r.get('BTC', np.nan),
                     'n': len(s)})
        w_prev = w
    d = pd.DataFrame(rows).set_index('t')
    d[['r', 'btc']].to_csv(os.path.join(os.path.dirname(__file__), 's05_tage.csv'))
    out = {}
    for name, (a, b) in PERIODS.items():
        x = d[(d.index >= a) & (d.index <= b)]
        m = x.r.mean(); sd = x.r.std()
        beta = np.cov(x.r, x.btc.fillna(0))[0, 1] / x.btc.fillna(0).var()
        eq = (1 + x.r).cumprod()
        out[name] = {'tage': len(x), 'coins_mittel': round(x.n.mean(), 1), 'ertrag_pa_pct': round(((1 + m) ** 365 - 1) * 100, 1),
                     't': round(m / (sd / np.sqrt(len(x))), 2), 'sharpe': round(m / sd * np.sqrt(365), 2),
                     'beta_btc': round(beta, 3), 'max_dd_pct': round(float((1 - eq / eq.cummax()).max() * 100), 1),
                     'umsatz_tag': round(x.turnover.mean(), 2)}
        print(name, json.dumps(out[name]))
    ok = all(out[p]['t'] >= 2 and out[p]['sharpe'] >= 1 and abs(out[p]['beta_btc']) <= 0.3
             for p in ('validierung', 'endtest'))
    out['bestanden'] = bool(ok)
    print('==>', 'BESTANDEN' if ok else 'NICHT BESTANDEN')
    json.dump(out, open(os.path.join(os.path.dirname(__file__), 's05_ergebnis.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
