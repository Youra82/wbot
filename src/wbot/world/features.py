# src/wbot/world/features.py
# Merkmale je Wetterstation, ausgerichtet auf den ENTSCHEIDUNGSTAG t (Entscheidung 00:05 UTC an Tag t).
# Ein Wert vom Datentag D darf fuer t verwendet werden, wenn D <= t - 1 - extra_lag (extra_lag je Quelle,
# siehe PRAEREGISTRIERUNG.md). Live und Backtest rufen exakt diese Funktionen auf.
import numpy as np
import pandas as pd

from wbot.world import sources as src

STATIONS = ['preis', 'makro', 'stablecoins', 'coinbase', 'korea', 'aufmerksamkeit', 'derivate']
FFILL_LIMIT = 5   # Wochenenden/Feiertage bei FRED


def _align(df, days, extra_lag=0):
    """Datentag -> Entscheidungstag: auffuellen, dann um 1 + extra_lag Tage nach hinten schieben."""
    full = pd.date_range(min(df.index.min(), days[0]), days[-1], freq='D')
    out = df.reindex(full).ffill(limit=FFILL_LIMIT).shift(1 + extra_lag)
    return out.reindex(days)


def _z(x, n):
    return (x - x.rolling(n, min_periods=n // 2).mean()) / x.rolling(n, min_periods=n // 2).std()


def _ret_z(close, n, vol_n=60):
    lr = np.log(close)
    sd = lr.diff().rolling(vol_n, min_periods=30).std()
    return (lr - lr.shift(n)) / (sd * np.sqrt(n))


def daily_index(px):
    return pd.date_range(px.index[0], px.index[-1] + pd.Timedelta(days=1), freq='D')


def atr14(px):
    h, l, c = px['high'], px['low'], px['close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


# ---------------- Stationen ----------------
def st_preis(coin, days):
    px = src.binance_klines(coin)
    c, h, l = px['close'], px['high'], px['low']
    atr = atr14(px)
    atrp = atr / c
    lc = np.log(c)
    f = pd.DataFrame(index=px.index)
    for n in (1, 3, 10, 30):
        f[f'r{n}'] = (lc - lc.shift(n)) / (atrp * np.sqrt(n))
    mx, mn = h.rolling(20).max(), l.rolling(20).min()
    f['pos20'] = (c - mn) / (mx - mn).clip(lower=1e-12) * 2 - 1
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f['volr'] = np.log(tr.rolling(5).mean() / tr.rolling(50).mean())
    f['dma50'] = (c - c.rolling(50).mean()) / atr
    return _align(f, days)


def st_makro(coin, days):
    nas, vix, y10 = (src.fred(k)['close'] for k in ('nasdaq', 'vix', 'us10y'))
    gold, eur = src.binance_spot_close('PAXG')['close'], src.binance_spot_close('EUR')['close']
    fred = pd.DataFrame({
        'nasdaq_r1': _ret_z(nas, 1), 'nasdaq_r5': _ret_z(nas, 5),
        'vix_z': _z(vix, 250), 'vix_d5': _z(vix - vix.shift(5), 250),
        'us10y_d5': _z(y10 - y10.shift(5), 250),
    })
    spot = pd.DataFrame({'gold_r5': _ret_z(gold, 5), 'eur_r5': _ret_z(eur, 5)})
    return pd.concat([_align(fred, days, extra_lag=2), _align(spot, days)], axis=1)


def st_stablecoins(coin, days):
    s = np.log(src.stablecoins()['supply'].replace(0, np.nan))
    f = pd.DataFrame({'stable_d7': _z(s - s.shift(7), 180), 'stable_d30': _z(s - s.shift(30), 180)})
    return _align(f, days, extra_lag=1)


def st_coinbase(coin, days):
    cols = {}
    for c in ('BTC', 'ETH'):
        prem = np.log(src.coinbase(c)['close'] / src.binance_spot_close(c)['close']).dropna()
        cols[f'cb_{c.lower()}_1'] = _z(prem, 90)
        if c == 'BTC':
            cols['cb_btc_7'] = _z(prem.rolling(7).mean(), 90)
    return _align(pd.DataFrame(cols), days)


def st_korea(coin, days):
    cols = {}
    for c in ('BTC', 'ETH'):
        kp = np.log(src.upbit(c)['close'] / src.binance_spot_close(c)['close']).dropna()
        dev = kp - kp.rolling(30, min_periods=15).mean()        # Wechselkursdrift faellt heraus
        cols[f'kr_{c.lower()}_1'] = dev / dev.rolling(90, min_periods=45).std()
        if c == 'BTC':
            d7 = dev.rolling(7).mean()
            cols['kr_btc_7'] = d7 / d7.rolling(90, min_periods=45).std()
    return _align(pd.DataFrame(cols), days)


def st_aufmerksamkeit(coin, days):
    own = np.log(src.wiki(coin)['views'].clip(lower=1))
    btc = np.log(src.wiki('BTC')['views'].clip(lower=1))
    f = pd.DataFrame({
        'wiki_z30': _z(own, 30),
        'wiki_ch7': (own.rolling(7).mean() - own.rolling(60, min_periods=30).mean()),
        'wiki_btc_z30': _z(btc, 30),
    })
    return _align(f, days, extra_lag=1)


def funding_daily(coin):
    fr = src.binance_funding(coin)['rate']
    return fr.groupby(fr.index.normalize()).mean() * 1e4


def st_derivate(coin, days):
    fd = funding_daily(coin)
    f = pd.DataFrame({'fund_1': fd, 'fund_7': fd.rolling(7).mean(), 'fund_z': _z(fd, 90)})
    return _align(f, days)


FUNCS = {'preis': st_preis, 'makro': st_makro, 'stablecoins': st_stablecoins, 'coinbase': st_coinbase,
         'korea': st_korea, 'aufmerksamkeit': st_aufmerksamkeit, 'derivate': st_derivate}


def coin_features(coin, days, stations=STATIONS):
    """Alle Merkmale eines Coins, Spalten 'station__merkmal', Index = Entscheidungstage."""
    parts = []
    for st in stations:
        f = FUNCS[st](coin, days)
        f.columns = [f'{st}__{c}' for c in f.columns]
        parts.append(f)
    return pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)


def station_columns(columns, station):
    return [c for c in columns if c.startswith(station + '__')]
