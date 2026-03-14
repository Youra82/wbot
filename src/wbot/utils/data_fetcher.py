# src/wbot/utils/data_fetcher.py
# OHLCV-Daten von Bitget via ccxt mit Cache-Unterstuetzung
import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

_TIMEFRAME_DURATION_MAP = {
    '1m':  timedelta(minutes=1),
    '5m':  timedelta(minutes=5),
    '15m': timedelta(minutes=15),
    '30m': timedelta(minutes=30),
    '1h':  timedelta(hours=1),
    '2h':  timedelta(hours=2),
    '4h':  timedelta(hours=4),
    '6h':  timedelta(hours=6),
    '1d':  timedelta(days=1),
    '1w':  timedelta(weeks=1),
}

_TIMEFRAME_MS_MAP = {
    '1m':  60_000,
    '5m':  300_000,
    '15m': 900_000,
    '30m': 1_800_000,
    '1h':  3_600_000,
    '2h':  7_200_000,
    '4h':  14_400_000,
    '6h':  21_600_000,
    '1d':  86_400_000,
    '1w':  604_800_000,
}

CACHE_DIR = os.path.join(PROJECT_ROOT, 'data', 'cache')


def _load_secret() -> Optional[dict]:
    """Laedt die secret.json und gibt den 'wbot'-Abschnitt zurueck."""
    secret_path = os.path.join(PROJECT_ROOT, 'secret.json')
    if not os.path.exists(secret_path):
        logger.debug("secret.json nicht gefunden - kein Exchange-Zugriff.")
        return None
    try:
        with open(secret_path, 'r') as f:
            secrets = json.load(f)
        wbot_secret = secrets.get('wbot', {})
        if not wbot_secret:
            logger.debug("Kein 'wbot'-Key in secret.json - kein Exchange-Zugriff.")
            return None
        return wbot_secret
    except Exception as e:
        logger.debug(f"Fehler beim Laden von secret.json: {e}")
        return None


def _create_exchange(wbot_secret: dict):
    """Erstellt eine ccxt Bitget-Exchange-Instanz."""
    try:
        import ccxt
        exchange = ccxt.bitget({
            'apiKey':   wbot_secret.get('api_key', ''),
            'secret':   wbot_secret.get('api_secret', ''),
            'password': wbot_secret.get('passphrase', ''),
            'options':  {'defaultType': 'swap'},
            'enableRateLimit': True,
        })
        return exchange
    except Exception as e:
        logger.warning(f"Exchange-Erstellung fehlgeschlagen: {e}")
        return None


def _symbol_to_filename(symbol: str) -> str:
    """Konvertiert Symbol-String zu sicherem Dateinamen (z.B. BTC/USDT:USDT -> BTCUSDTUSDT)."""
    return symbol.replace('/', '').replace(':', '')


def fetch_ohlcv(
    exchange,
    symbol: str,
    timeframe: str,
    start_dt_str: str,
    end_dt_str: str
) -> pd.DataFrame:
    """
    Laedt OHLCV-Daten paginiert von Bitget (max 1000 Kerzen pro Request).

    Args:
        exchange: ccxt Exchange-Instanz
        symbol: Handelspaar (z.B. "BTC/USDT:USDT")
        timeframe: Zeitrahmen (z.B. "1d", "4h")
        start_dt_str: Startdatum als String "YYYY-MM-DD"
        end_dt_str: Enddatum als String "YYYY-MM-DD"

    Returns:
        DataFrame mit [open, high, low, close, volume], Index=timestamp (UTC)
    """
    try:
        start_dt = datetime.strptime(start_dt_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_dt   = datetime.strptime(end_dt_str,   '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_dt   = end_dt + timedelta(days=1)  # Enddatum inklusive

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms   = int(end_dt.timestamp()   * 1000)
        tf_ms    = _TIMEFRAME_MS_MAP.get(timeframe, 86_400_000)

        all_ohlcv = []
        current_since = start_ms
        max_per_request = 1000

        logger.info(f"Lade Daten paginiert: {symbol} ({timeframe}) von {start_dt_str} bis {end_dt_str}")

        while current_since < end_ms:
            try:
                batch = exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=max_per_request
                )
            except Exception as e:
                logger.warning(f"fetch_ohlcv Batch-Fehler: {e}")
                break

            if not batch:
                break

            # Filtere Kerzen die nach end_ms liegen
            filtered = [c for c in batch if c[0] < end_ms]
            all_ohlcv.extend(filtered)

            if len(batch) < max_per_request:
                break  # Keine weiteren Daten verfuegbar

            # Naechste Seite
            last_ts = batch[-1][0]
            current_since = last_ts + tf_ms

            if last_ts >= end_ms - tf_ms:
                break

        if not all_ohlcv:
            logger.warning(f"Keine Daten empfangen fuer {symbol} ({timeframe}).")
            return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.set_index('timestamp')

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna()
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]

        logger.info(f"Daten geladen: {len(df)} Kerzen | {df.index[0]} bis {df.index[-1]}")
        return df

    except Exception as e:
        logger.error(f"fetch_ohlcv fehlgeschlagen fuer {symbol} ({timeframe}): {e}", exc_info=True)
        return pd.DataFrame()


def load_data(
    symbol: str,
    timeframe: str,
    start_date_str: str,
    end_date_str: str
) -> pd.DataFrame:
    """
    Laedt OHLCV-Daten mit Cache-Unterstuetzung.

    Cache in data/cache/{symbol_filename}_{timeframe}.csv
    Puffer: 30 Tage extra fuer Indikator-Berechnungen.
    Fallback: wenn kein secret.json -> return pd.DataFrame()

    Args:
        symbol: Handelspaar (z.B. "BTC/USDT:USDT")
        timeframe: Zeitrahmen (z.B. "1d")
        start_date_str: Startdatum "YYYY-MM-DD"
        end_date_str: Enddatum "YYYY-MM-DD"

    Returns:
        DataFrame mit [open, high, low, close, volume], Index=timestamp
    """
    wbot_secret = _load_secret()
    if wbot_secret is None:
        logger.warning("Kein secret.json / wbot-Key gefunden. Gebe leeres DataFrame zurueck.")
        return pd.DataFrame()

    # Puffer fuer Indikator-Berechnungen (30 Tage extra)
    try:
        start_dt_buffered = datetime.strptime(start_date_str, '%Y-%m-%d') - timedelta(days=30)
        start_with_buffer = start_dt_buffered.strftime('%Y-%m-%d')
    except ValueError:
        start_with_buffer = start_date_str

    os.makedirs(CACHE_DIR, exist_ok=True)
    symbol_filename = _symbol_to_filename(symbol)
    cache_file = os.path.join(CACHE_DIR, f"{symbol_filename}_{timeframe}.csv")

    # Versuche Cache zu lesen
    cached_df = pd.DataFrame()
    if os.path.exists(cache_file):
        try:
            cached_df = pd.read_csv(cache_file, index_col='timestamp', parse_dates=True)
            # Stelle sicher, dass Index timezone-aware ist
            if cached_df.index.tzinfo is None:
                cached_df.index = cached_df.index.tz_localize('UTC')
            logger.debug(f"Cache geladen: {cache_file} ({len(cached_df)} Kerzen)")
        except Exception as e:
            logger.warning(f"Cache-Lese-Fehler: {e}. Lade neu.")
            cached_df = pd.DataFrame()

    # Prueffe ob Cache die benoetigten Daten enthaelt
    try:
        need_start = pd.Timestamp(start_with_buffer, tz='UTC')
        need_end   = pd.Timestamp(end_date_str, tz='UTC') + timedelta(days=1)
    except Exception:
        need_start = None
        need_end   = None

    cache_valid = (
        not cached_df.empty
        and need_start is not None
        and need_end is not None
        and cached_df.index[0] <= need_start
        and cached_df.index[-1] >= need_end - timedelta(days=1)
    )

    if cache_valid:
        logger.info(f"Nutze Cache fuer {symbol} ({timeframe}): {len(cached_df)} Kerzen")
        df = cached_df
    else:
        # Frische Daten laden
        exchange = _create_exchange(wbot_secret)
        if exchange is None:
            if not cached_df.empty:
                logger.warning("Exchange nicht verfuegbar. Nutze vorhandenen Cache.")
                df = cached_df
            else:
                return pd.DataFrame()
        else:
            df = fetch_ohlcv(exchange, symbol, timeframe, start_with_buffer, end_date_str)
            if df.empty:
                if not cached_df.empty:
                    logger.warning("Keine neuen Daten. Nutze Cache-Fallback.")
                    df = cached_df
                else:
                    return pd.DataFrame()

            # Cache aktualisieren
            try:
                df.to_csv(cache_file, index=True, index_label='timestamp')
                logger.debug(f"Cache geschrieben: {cache_file}")
            except Exception as e:
                logger.warning(f"Cache-Schreib-Fehler: {e}")

    # Filtere auf gewuenschten Zeitraum (inkl. Puffer fuer Indikatoren)
    if not df.empty and need_start is not None:
        df = df[df.index >= need_start]
        if need_end is not None:
            df = df[df.index < need_end]

    return df


def is_new_candle(df: pd.DataFrame, timeframe: str) -> bool:
    """
    Prueft ob eine neue Kerze seit dem letzten Run geoeffnet hat.

    Args:
        df: OHLCV-DataFrame mit Zeitstempel-Index
        timeframe: Zeitrahmen (z.B. "1d", "4h")

    Returns:
        True wenn neue Kerze seit < 1 Kerzen-Dauer geoeffnet
    """
    if df is None or len(df) == 0:
        return False

    try:
        candle_duration = _TIMEFRAME_DURATION_MAP.get(timeframe, timedelta(days=1))
        now_utc = datetime.now(timezone.utc)

        last_candle_time = df.index[-1]
        if hasattr(last_candle_time, 'tzinfo') and last_candle_time.tzinfo is None:
            last_candle_time = last_candle_time.replace(tzinfo=timezone.utc)

        next_candle_time = last_candle_time + candle_duration
        time_into_new_candle = now_utc - next_candle_time
        tolerance = candle_duration * 0.10

        is_new = timedelta(0) <= time_into_new_candle <= tolerance

        logger.debug(
            f"Neue Kerze? {is_new} | letzte={last_candle_time} | "
            f"naechste={next_candle_time} | jetzt={now_utc}"
        )
        return is_new

    except Exception as e:
        logger.warning(f"Neue-Kerze-Pruefung fehlgeschlagen: {e}")
        return False
