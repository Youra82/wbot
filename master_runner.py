# master_runner.py
# wbot QGRS - Orchestriert alle aktiven Strategien
# Wird per Cronjob aufgerufen (z.B. alle 4h fuer 1d-Timeframe)
import json
import subprocess
import sys
import os
import time
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

log_dir = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'master_runner.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)


def main():
    settings_file = os.path.join(SCRIPT_DIR, 'settings.json')
    secret_file = os.path.join(SCRIPT_DIR, 'secret.json')
    bot_runner_script = os.path.join(SCRIPT_DIR, 'src', 'wbot', 'strategy', 'run.py')

    # Python-Interpreter in der venv
    python_executable = os.path.join(SCRIPT_DIR, '.venv', 'bin', 'python3')
    if not os.path.exists(python_executable):
        logging.critical(f"Python-Interpreter nicht gefunden: {python_executable}")
        return

    logging.info("=" * 60)
    logging.info("wbot QGRS Master Runner - Cronjob-basiert")
    logging.info("=" * 60)

    try:
        with open(settings_file, 'r') as f:
            settings = json.load(f)

        with open(secret_file, 'r') as f:
            secrets = json.load(f)

        if not secrets.get('wbot'):
            logging.critical("Kein 'wbot'-Account in secret.json gefunden.")
            return

        live_settings = settings.get('live_trading_settings', {})
        strategy_list = live_settings.get('active_strategies', [])

        if not strategy_list:
            logging.warning("Keine aktiven Strategien in settings.json konfiguriert.")
            return

        logging.info(f"Gefundene Strategien: {len(strategy_list)}")

        for strategy_info in strategy_list:
            if not isinstance(strategy_info, dict):
                continue
            if not strategy_info.get('active', False):
                logging.info(f"Strategie inaktiv (skip): {strategy_info}")
                continue

            symbol = strategy_info.get('symbol')
            timeframe = strategy_info.get('timeframe', '1d')

            if not symbol:
                logging.warning(f"Unvollstaendige Strategie-Info: {strategy_info}")
                continue

            logging.info(f"Starte wbot fuer: {symbol} ({timeframe})")
            command = [
                python_executable,
                bot_runner_script,
                "--symbol", symbol,
                "--timeframe", timeframe,
            ]

            env = os.environ.copy()
            env['PYTHONPATH'] = os.path.join(SCRIPT_DIR, 'src')

            try:
                process = subprocess.Popen(command, env=env)
                logging.info(f"Prozess gestartet fuer {symbol}_{timeframe} (PID: {process.pid}).")
                time.sleep(2)
            except Exception as e:
                logging.error(f"Fehler beim Starten des Prozesses fuer {symbol}_{timeframe}: {e}")

    except FileNotFoundError as e:
        logging.critical(f"Datei nicht gefunden: {e}")
    except json.JSONDecodeError as e:
        logging.critical(f"JSON-Parsing-Fehler: {e}")
    except Exception as e:
        logging.critical(f"Unerwarteter Fehler im Master Runner: {e}", exc_info=True)


if __name__ == "__main__":
    main()
