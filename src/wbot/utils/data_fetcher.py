# src/wbot/utils/data_fetcher.py
# OHLCV-Daten von Bitget via ccxt
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
    '1m': timedelta(minutes=1),
    '5m': timedelta(minutes=5),
    '15m': timedelta(minutes=15),
    '30m': timedelta(minutes=30),
    '1h': timedelta(hours=1),
    '4h': timedelta(hours=4),
    '1d': timedelta(days=1),
    '1w': timedelta(weeks=1),
}


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
            'apiKey': wbot_secret.get('api_key', ''),
            'secret': wbot_secret.get('api_secret', ''),
            'password': wbot_secret.get('passphrase', ''),
            'options': {'defaultType': 'swap'},
            'enableRateLimit': True,
        })
        return exchange
    except Exception as e:
        logger.warning(f"Exchange-Erstellung fehlgeschlagen: {e}")
        return None


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    limit: int = 300
) -> Optional[pd.DataFrame]:
    """
    Laedt OHLCV-Daten von Bitget via ccxt.

    Falls keine API-Keys verfuegbar sind, wird None zurueckgegeben.

    Args:
        symbol: Handelspaar (z.B. "BTC/USDT:USDT")
        timeframe: Zeitrahmen (z.B. "1d", "4h")
        limit: Anzahl Kerzen

    Returns:
        DataFrame mit [timestamp, open, high, low, close, volume] oder None
    """
    wbot_secret = _load_secret()

    if wbot_secret is None:
        logger.warning("Keine API-Keys verfuegbar. Gebe None zurueck.")
        return None

    exchange = _create_exchange(wbot_secret)
    if exchange is None:
        return None

    try:
        logger.info(f"Lade {limit} {timeframe}-Kerzen fuer {symbol} von Bitget...")

        ohlcv_raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

        if not ohlcv_raw:
            logger.warning(f"Keine Daten von Bitget erhalten fuer {symbol} ({timeframe}).")
            return None

        df = pd.DataFrame(
            ohlcv_raw,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        # Timestamp in datetime konvertieren
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.set_index('timestamp')

        # Sicherstellen dass alle Werte numerisch sind
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Entferne NaN-Zeilen
        df = df.dropna()

        # Sortiere nach Zeit aufsteigend
        df = df.sort_index()

        logger.info(f"OHLCV geladen: {len(df)} Kerzen | {df.index[0]} bis {df.index[-1]}")
        return df

    except Exception as e:
        logger.error(f"OHLCV-Ladung fehlgeschlagen fuer {symbol} ({timeframe}): {e}", exc_info=True)
        return None


def is_new_candle(df: pd.DataFrame, timeframe: str) -> bool:
    """
    Prueft ob eine neue Kerze seit dem letzten Run geoeffnet hat.

    Methode: Vergleicht Zeitstempel der letzten Kerze mit der erwarteten
    Eroeffnungszeit basierend auf dem Timeframe.

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

        # Zeitstempel der letzten Kerze
        last_candle_time = df.index[-1]
        if hasattr(last_candle_time, 'tzinfo') and last_candle_time.tzinfo is None:
            last_candle_time = last_candle_time.replace(tzinfo=timezone.utc)

        # Naechste erwartete Kerzen-Eroeffnung
        next_candle_time = last_candle_time + candle_duration

        # Pruefe ob wir innerhalb des ersten 10% der neuen Kerze sind
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
