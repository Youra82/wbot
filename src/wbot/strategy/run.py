# src/wbot/strategy/run.py
# wbot QGRS - Live-Trading Entry Point (Multi-Symbol/Timeframe mit Config-Dateien)
import os
import sys
import json
import logging
import argparse
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))


def setup_logging(symbol: str, timeframe: str) -> logging.Logger:
    """Konfiguriert Logging fuer eine spezifische Strategie-Instanz."""
    safe_filename = f"{symbol.replace('/', '').replace(':', '')}_{timeframe}"
    log_dir  = os.path.join(PROJECT_ROOT, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'wbot_{safe_filename}.log')

    logger_name = f'wbot_{safe_filename}'
    logger      = logging.getLogger(logger_name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(
            f'%(asctime)s [wbot|{safe_filename}] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        ))
        logger.addHandler(ch)
        logger.propagate = False

    return logger


def send_telegram(telegram_config: dict, message: str, logger: logging.Logger):
    """Sendet eine Telegram-Benachrichtigung."""
    try:
        import requests
        token   = telegram_config.get('token') or telegram_config.get('bot_token')
        chat_id = telegram_config.get('chat_id')
        if not token or not chat_id:
            return
        url  = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, data={'chat_id': chat_id, 'text': message}, timeout=10)
        if not resp.ok:
            logger.warning(f"Telegram-Sende-Fehler: {resp.text}")
    except Exception as e:
        logger.warning(f"Telegram-Benachrichtigung fehlgeschlagen: {e}")


def load_strategy_config(symbol: str, timeframe: str) -> dict:
    """
    Laedt die optimierte Config-Datei fuer eine Strategie.

    Sucht nach: src/wbot/strategy/configs/config_{symbol}_{timeframe}.json

    Returns:
        Config-Dict mit market/physics/strategy/risk Sektionen
        oder None wenn nicht gefunden
    """
    safe_name   = f"{symbol.replace('/', '').replace(':', '')}_{timeframe}"
    config_dir  = os.path.join(PROJECT_ROOT, 'src', 'wbot', 'strategy', 'configs')
    config_path = os.path.join(config_dir, f"config_{safe_name}.json")

    if not os.path.exists(config_path):
        return None

    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception as e:
        return None


def is_entry_candle(timeframe: str, logger: logging.Logger) -> bool:
    """
    Prueft ob eine neue Kerze geoeffnet hat (Entry nur bei neuer Kerze).
    """
    now_utc = datetime.now(timezone.utc)

    timeframe_lower = timeframe.lower()

    if timeframe_lower == '1d':
        is_new = now_utc.hour < 4
        if not is_new:
            logger.info(
                f"Keine neue 1d-Kerze (UTC-Stunde={now_utc.hour}, Entry nur bei 0-3 UTC). "
                f"Nur Position-Check."
            )
        return is_new
    elif timeframe_lower == '4h':
        return now_utc.hour % 4 == 0
    elif timeframe_lower == '6h':
        return now_utc.hour % 6 == 0
    elif timeframe_lower == '1h':
        return True
    else:
        return True


def main():
    parser = argparse.ArgumentParser(description="wbot QGRS Live-Trading")
    parser.add_argument('--symbol',    required=True, type=str,
                        help="Handelspaar (z.B. BTC/USDT:USDT)")
    parser.add_argument('--timeframe', default='1d', type=str,
                        help="Zeitrahmen (default: 1d)")
    args = parser.parse_args()

    symbol    = args.symbol
    timeframe = args.timeframe

    logger = setup_logging(symbol, timeframe)
    logger.info("=" * 60)
    logger.info(f"wbot QGRS gestartet | Symbol={symbol} | TF={timeframe}")
    logger.info("=" * 60)

    # --- Settings laden ---
    settings_path = os.path.join(PROJECT_ROOT, 'settings.json')
    secret_path   = os.path.join(PROJECT_ROOT, 'secret.json')

    try:
        with open(settings_path) as f:
            settings = json.load(f)
    except FileNotFoundError:
        logger.critical(f"settings.json nicht gefunden: {settings_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.critical(f"Fehler beim Parsen von settings.json: {e}")
        sys.exit(1)

    try:
        with open(secret_path) as f:
            secrets = json.load(f)
    except FileNotFoundError:
        logger.critical(f"secret.json nicht gefunden: {secret_path}")
        sys.exit(1)

    wbot_secret    = secrets.get('wbot', {})
    telegram_config = secrets.get('telegram', {})

    # --- Strategie-Config laden (optimierte Config-Datei hat Vorrang) ---
    strategy_config = load_strategy_config(symbol, timeframe)

    if strategy_config:
        logger.info(f"Optimierte Config geladen fuer {symbol} ({timeframe})")
        physics_cfg  = strategy_config.get('physics', {})
        signal_cfg   = {
            **strategy_config.get('strategy', {}),
            **strategy_config.get('risk', {}),
        }
        n_simulations = 10000
        n_steps       = 288
    else:
        logger.warning(
            f"Keine optimierte Config fuer {symbol} ({timeframe}) gefunden. "
            f"Nutze Default-Parameter aus settings.json."
        )
        # Fallback auf settings.json
        physics_cfg   = settings.get('physics', {})
        signal_cfg    = settings.get('strategy', {})
        n_simulations = signal_cfg.get('n_simulations', 10000)
        n_steps       = signal_cfg.get('n_steps_intraday', 288)
        # Baue config-Dict fuer compute_market_state
        strategy_config = {
            'physics':  physics_cfg,
            'strategy': settings.get('strategy', {}),
            'risk':     {},
        }

    # --- Imports ---
    from wbot.utils.data_fetcher import fetch_ohlcv, is_new_candle
    from wbot.model.market_state import compute_market_state
    from wbot.model.monte_carlo import run_monte_carlo
    from wbot.forecast.range_forecast import compute_range_forecast, format_forecast_table
    from wbot.strategy.signal_logic import generate_signal
    from wbot.utils.trade_manager import get_open_positions, place_entry_order

    # --- Daten laden ---
    logger.info(f"Lade OHLCV-Daten von Bitget fuer {symbol} ({timeframe})...")
    df = fetch_ohlcv(symbol, timeframe, limit=300)

    if df is None or len(df) < 50:
        logger.critical(
            f"Nicht genug Daten geladen (erhalten: {len(df) if df is not None else 0})."
        )
        sys.exit(1)

    logger.info(f"Daten geladen: {len(df)} Kerzen.")
    current_price = float(df['close'].iloc[-1])
    logger.info(f"Aktueller Preis: {current_price:.4f}")

    # --- Exchange initialisieren ---
    exchange = None
    try:
        import ccxt
        api_key    = wbot_secret.get('api_key', '')
        api_secret = wbot_secret.get('api_secret', '')
        passphrase = wbot_secret.get('passphrase', '')

        if api_key and api_secret:
            exchange = ccxt.bitget({
                'apiKey':   api_key,
                'secret':   api_secret,
                'password': passphrase,
                'options':  {'defaultType': 'swap'},
            })
            logger.info("Exchange-Verbindung hergestellt.")
        else:
            logger.warning("Keine API-Keys in secret.json - Simulation-Modus.")
    except Exception as e:
        logger.warning(f"Exchange-Initialisierung fehlgeschlagen: {e}. Simulation-Modus.")

    # --- Bestehende Positionen pruefen ---
    open_positions = []
    if exchange is not None:
        try:
            open_positions = get_open_positions(exchange, symbol)
            if open_positions:
                logger.info(f"Offene Positionen: {len(open_positions)}")
        except Exception as e:
            logger.warning(f"Positions-Abfrage fehlgeschlagen: {e}")

    # --- MarketState berechnen ---
    logger.info("Berechne MarketState (Physics-Features)...")
    try:
        state = compute_market_state(df, config=strategy_config)
    except Exception as e:
        logger.critical(f"MarketState-Berechnung fehlgeschlagen: {e}", exc_info=True)
        sys.exit(1)

    # --- Monte-Carlo-Simulation ---
    logger.info(f"Starte Monte-Carlo ({n_simulations} Simulationen)...")
    try:
        sim_result = run_monte_carlo(state, n_simulations=n_simulations, n_steps=n_steps)
    except Exception as e:
        logger.critical(f"Monte-Carlo fehlgeschlagen: {e}", exc_info=True)
        sys.exit(1)

    # --- Range-Forecast ---
    try:
        forecast        = compute_range_forecast(sim_result, current_price)
        forecast_table  = format_forecast_table(forecast)
        logger.info(forecast_table)
    except Exception as e:
        logger.critical(f"Range-Forecast fehlgeschlagen: {e}", exc_info=True)
        sys.exit(1)

    # --- Signal generieren ---
    try:
        signal = generate_signal(forecast, state, current_price, signal_cfg)
        logger.info(
            f"Signal: {signal.action} | Confidence={signal.confidence:.2f} | "
            f"Reason={signal.reason}"
        )
    except Exception as e:
        logger.critical(f"Signal-Generierung fehlgeschlagen: {e}", exc_info=True)
        sys.exit(1)

    # --- Trade-Execution ---
    if signal.action == "wait":
        logger.info("Kein Trade-Signal. Beende Lauf.")
        send_telegram(
            telegram_config,
            f"wbot QGRS | {symbol} ({timeframe})\n"
            f"Signal: WARTEN\n"
            f"E[Range]: {forecast.expected_range_pct:.2f}%\n"
            f"Grund: {signal.reason}",
            logger
        )
        logger.info(">>> wbot Lauf abgeschlossen (kein Trade) <<<")
        return

    if open_positions:
        logger.info("Bereits offene Position vorhanden - kein neuer Entry.")
        logger.info(">>> wbot Lauf abgeschlossen (Position bereits offen) <<<")
        return

    if not is_entry_candle(timeframe, logger):
        logger.info("Kein Entry - warte auf neue Kerze.")
        logger.info(">>> wbot Lauf abgeschlossen (warte auf neue Kerze) <<<")
        return

    if exchange is not None:
        try:
            leverage = signal_cfg.get('leverage', 10)

            # Kapital vom Exchange abfragen
            try:
                balance_info  = exchange.fetch_balance()
                capital_usdt  = float(
                    balance_info.get('USDT', {}).get('free', 1000.0)
                )
            except Exception:
                capital_usdt = 1000.0
                logger.warning("Kapital-Abfrage fehlgeschlagen, nutze 1000 USDT als Fallback.")

            from wbot.utils.trade_manager import calculate_position_size
            size_usdt = calculate_position_size(
                capital=capital_usdt,
                risk_pct=signal.position_size_pct,
                entry=signal.entry_price,
                stop_loss=signal.stop_loss,
                leverage=leverage
            )

            side = 'buy' if 'long' in signal.action else 'sell'

            logger.info(
                f"Platziere Order: {side} | entry={signal.entry_price:.4f} | "
                f"sl={signal.stop_loss:.4f} | tp={signal.take_profit:.4f} | "
                f"size={size_usdt:.2f} USDT"
            )

            order = place_entry_order(
                exchange=exchange,
                symbol=symbol,
                side=side,
                size_usdt=size_usdt,
                entry_price=signal.entry_price,
                sl_price=signal.stop_loss,
                tp_price=signal.take_profit,
                leverage=leverage
            )

            if order:
                logger.info(f"Order platziert: {order}")
                send_telegram(
                    telegram_config,
                    f"wbot QGRS | {symbol} ({timeframe})\n"
                    f"Signal: {signal.action.upper()}\n"
                    f"Entry: {signal.entry_price:.4f}\n"
                    f"SL: {signal.stop_loss:.4f}\n"
                    f"TP: {signal.take_profit:.4f}\n"
                    f"Confidence: {signal.confidence:.2f}\n"
                    f"E[Range]: {forecast.expected_range_pct:.2f}%\n"
                    f"Grund: {signal.reason}",
                    logger
                )
            else:
                logger.error("Order-Platzierung lieferte kein Ergebnis.")

        except Exception as e:
            logger.error(f"Trade-Execution fehlgeschlagen: {e}", exc_info=True)
            send_telegram(telegram_config, f"wbot QGRS FEHLER | {symbol}: {e}", logger)
    else:
        logger.info(
            f"[Simulation] Signal: {signal.action} | "
            f"entry={signal.entry_price:.4f} | "
            f"sl={signal.stop_loss:.4f} | "
            f"tp={signal.take_profit:.4f}"
        )

    logger.info(">>> wbot Lauf abgeschlossen <<<")


if __name__ == "__main__":
    main()
