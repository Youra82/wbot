# src/wbot/world/sources.py
# Rohdaten aller "Wetterstationen" laden und als Tages-CSV cachen (inkrementell).
# Jede Reihe ist nach DATENTAG indiziert (UTC-Datum, tz-naiv). Welche Tage zum Entscheidungszeitpunkt
# schon bekannt sind, regelt features.py ueber die Verzoegerung je Station.
import os
import time
import logging
import requests
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
WORLD_DIR = os.path.join(PROJECT_ROOT, 'data', 'world')
START = pd.Timestamp('2020-01-01')
UA = {'User-Agent': 'wbot-research/1.0 (python-requests; non-commercial research)'}

COINS = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'BNB']
FRED = {'nasdaq': 'NASDAQCOM', 'vix': 'VIXCLS', 'us10y': 'DGS10'}
SPOT_EXTRA = {'gold': 'PAXG', 'eur': 'EUR'}   # Binance-Spot gegen USDT, 24/7
WIKI = {'BTC': 'Bitcoin', 'ETH': 'Ethereum', 'SOL': 'Solana_(blockchain_platform)', 'XRP': 'XRP_Ledger',
        'DOGE': 'Dogecoin', 'BNB': 'Binance'}
UPBIT = {'BTC': 'KRW-BTC', 'ETH': 'KRW-ETH', 'USDT': 'KRW-USDT'}
COINBASE = {'BTC': 'BTC-USD', 'ETH': 'ETH-USD'}


def _get(url, params=None, tries=4):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (k + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if k == tries - 1:
                raise
            logger.warning(f"HTTP-Fehler {url}: {e} (Versuch {k + 1})")
            time.sleep(2 * (k + 1))


def _cached(name, fetch, refetch_days=10, max_age_h=6):
    """Laedt <name>.csv, holt ab (letzter Tag - refetch_days) neu und fuegt zusammen."""
    os.makedirs(WORLD_DIR, exist_ok=True)
    path = os.path.join(WORLD_DIR, f'{name}.csv')
    old = None
    if os.path.exists(path):
        old = pd.read_csv(path, index_col=0, parse_dates=True)
        if time.time() - os.path.getmtime(path) < max_age_h * 3600:
            return old
    since = START if old is None or old.empty else old.index[-1] - pd.Timedelta(days=refetch_days)
    new = fetch(since)
    if old is not None and not old.empty:
        new = pd.concat([old[old.index < since], new]) if new is not None and not new.empty else old
    new = new[~new.index.duplicated(keep='last')].sort_index()
    new.to_csv(path)
    return new


def _day(ts_ms):
    return pd.to_datetime(ts_ms, unit='ms').normalize()


# ---------- Binance Futures: Tageskerzen + Funding ----------
def binance_klines(coin):
    def fetch(since):
        rows, cur = [], int(since.timestamp() * 1000)
        while True:
            b = _get('https://fapi.binance.com/fapi/v1/klines',
                     {'symbol': f'{coin}USDT', 'interval': '1d', 'startTime': cur, 'limit': 1500})
            if not b:
                break
            rows += b
            cur = b[-1][0] + 86_400_000
            if len(b) < 1500:
                break
        df = pd.DataFrame([[float(x) for x in r[1:6]] for r in rows],
                          index=_day([r[0] for r in rows]), columns=['open', 'high', 'low', 'close', 'volume'])
        return df
    df = _cached(f'px_{coin}', fetch)
    # laufende (unvollstaendige) Tageskerze nie verwenden
    return df[df.index < pd.Timestamp.utcnow().tz_localize(None).normalize()]


def binance_funding(coin):
    """Funding-Ereignisse (Zeitpunkt, Rate). Index = genauer Zeitstempel."""
    def fetch(since):
        rows, cur = [], int(since.timestamp() * 1000)
        while True:
            b = _get('https://fapi.binance.com/fapi/v1/fundingRate',
                     {'symbol': f'{coin}USDT', 'startTime': cur, 'limit': 1000})
            if not b:
                break
            rows += b
            cur = b[-1]['fundingTime'] + 1
            if len(b) < 1000:
                break
        return pd.DataFrame({'rate': [float(r['fundingRate']) for r in rows]},
                            index=pd.to_datetime([r['fundingTime'] for r in rows], unit='ms'))
    return _cached(f'funding_{coin}', fetch, max_age_h=1)


def binance_spot_close(coin):
    """Binance-Spot <coin>USDT Tagesschluss (auch PAXG = Gold, EUR = Euro/Dollar)."""
    def fetch(since):
        rows, cur = [], int(since.timestamp() * 1000)
        while True:
            b = _get('https://api.binance.com/api/v3/klines',
                     {'symbol': f'{coin}USDT', 'interval': '1d', 'startTime': cur, 'limit': 1000})
            if not b:
                break
            rows += b
            cur = b[-1][0] + 86_400_000
            if len(b) < 1000:
                break
        return pd.DataFrame({'close': [float(r[4]) for r in rows]}, index=_day([r[0] for r in rows]))
    return _cached(f'spot_{coin}', fetch)


# ---------- FRED (US-Notenbank St. Louis): Nasdaq, VIX, 10J-Zins ----------
def fred(key):
    def fetch(since):
        r = requests.get('https://fred.stlouisfed.org/graph/fredgraph.csv', params={'id': FRED[key]},
                         headers=UA, timeout=60)
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text), index_col=0, parse_dates=True, na_values='.')
        df.columns = ['close']
        return df.dropna().loc[since:]
    return _cached(f'fred_{key}', fetch)


# ---------- DefiLlama Stablecoin-Gesamtmenge ----------
def stablecoins():
    def fetch(since):
        j = _get('https://stablecoins.llama.fi/stablecoincharts/all')
        s = pd.Series({pd.to_datetime(int(r['date']), unit='s').normalize():
                       r.get('totalCirculatingUSD', {}).get('peggedUSD', np.nan) for r in j})
        return s.to_frame('supply').sort_index().loc[since:]
    return _cached('stablecoins', fetch)


# ---------- Coinbase ----------
def coinbase(coin):
    def fetch(since):
        out, cur, end = [], since, pd.Timestamp.utcnow().tz_localize(None)
        while cur < end:
            nxt = min(cur + pd.Timedelta(days=290), end)
            b = _get(f'https://api.exchange.coinbase.com/products/{COINBASE[coin]}/candles',
                     {'granularity': 86400, 'start': cur.strftime('%Y-%m-%dT%H:%M:%SZ'),
                      'end': nxt.strftime('%Y-%m-%dT%H:%M:%SZ')})
            out += b
            cur = nxt
            time.sleep(0.3)
        df = pd.DataFrame({'close': [float(r[4]) for r in out]},
                          index=pd.to_datetime([r[0] for r in out], unit='s').normalize())
        return df[~df.index.duplicated()].sort_index()
    return _cached(f'coinbase_{coin}', fetch)


# ---------- Upbit (Korea) ----------
def upbit(coin):
    def fetch(since):
        out, to = [], None
        while True:
            p = {'market': UPBIT[coin], 'count': 200}
            if to:
                p['to'] = to
            b = _get('https://api.upbit.com/v1/candles/days', p)
            if not b:
                break
            out += b
            first = pd.Timestamp(b[-1]['candle_date_time_utc'])
            if first <= since or len(b) < 200:
                break
            to = first.strftime('%Y-%m-%dT%H:%M:%SZ')
            time.sleep(0.15)
        df = pd.DataFrame({'close': [float(r['trade_price']) for r in out]},
                          index=pd.to_datetime([r['candle_date_time_utc'] for r in out]).normalize())
        return df[~df.index.duplicated()].sort_index().loc[since:]
    return _cached(f'upbit_{coin}', fetch)


# ---------- Wikipedia-Aufrufe ----------
def wiki(coin):
    def fetch(since):
        end = pd.Timestamp.utcnow().tz_localize(None)
        j = _get('https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/'
                 f'{WIKI[coin]}/daily/{since:%Y%m%d}/{end:%Y%m%d}')
        items = j.get('items', [])
        return pd.DataFrame({'views': [float(i['views']) for i in items]},
                            index=pd.to_datetime([i['timestamp'][:8] for i in items]))
    return _cached(f'wiki_{coin}', fetch)


def refresh_all():
    """Alle Quellen aktualisieren (Live: einmal taeglich). Fehler einzelner Quellen sind nicht fatal."""
    jobs = [(f'px_{c}', lambda c=c: binance_klines(c)) for c in COINS]
    jobs += [(f'funding_{c}', lambda c=c: binance_funding(c)) for c in COINS]
    jobs += [(f'spot_{c}', lambda c=c: binance_spot_close(c)) for c in COINBASE]
    jobs += [(f'spot_{c}', lambda c=c: binance_spot_close(c)) for c in SPOT_EXTRA.values()]
    jobs += [(f'fred_{k}', lambda k=k: fred(k)) for k in FRED]
    jobs += [('stablecoins', stablecoins)]
    jobs += [(f'coinbase_{c}', lambda c=c: coinbase(c)) for c in COINBASE]
    jobs += [(f'upbit_{c}', lambda c=c: upbit(c)) for c in UPBIT]
    jobs += [(f'wiki_{c}', lambda c=c: wiki(c)) for c in WIKI]
    failed = []
    for name, fn in jobs:
        try:
            df = fn()
            logger.info(f"{name}: {len(df)} Zeilen bis {df.index[-1].date() if len(df) else '-'}")
        except Exception as e:
            logger.error(f"{name} fehlgeschlagen: {e}")
            failed.append(name)
    return failed


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    print('Fehlgeschlagen:', refresh_all())
