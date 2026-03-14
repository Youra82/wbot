#!/usr/bin/env python3
"""
auto_optimizer_scheduler.py  —  wbot QGRS

Prueft bei jedem Aufruf ob eine Optimierung faellig ist und fuehrt
sie direkt via Python aus. Sendet Telegram-Benachrichtigungen bei Start und Ende.

Aufruf:
  python3 auto_optimizer_scheduler.py           # normale Pruefung
  python3 auto_optimizer_scheduler.py --force   # sofort erzwingen
"""

import os
import sys
import json
import time
import subprocess
import argparse
from datetime import datetime, date, timedelta

PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

CACHE_DIR             = os.path.join(PROJECT_ROOT, 'data', 'cache')
LOG_DIR               = os.path.join(PROJECT_ROOT, 'logs')
SETTINGS_FILE         = os.path.join(PROJECT_ROOT, 'settings.json')
OPTIMIZER_SCRIPT      = os.path.join(PROJECT_ROOT, 'src', 'wbot', 'analysis', 'optimizer.py')
SECRET_FILE           = os.path.join(PROJECT_ROOT, 'secret.json')
LAST_RUN_FILE         = os.path.join(CACHE_DIR, '.last_optimization_run')
IN_PROGRESS_FILE      = os.path.join(CACHE_DIR, '.optimization_in_progress')
TRIGGER_LOG           = os.path.join(LOG_DIR, 'auto_optimizer_trigger.log')
OPTIMIZER_RESULTS_FILE = os.path.join(
    PROJECT_ROOT, 'artifacts', 'results', 'last_optimizer_run.json'
)

# Lookback je Timeframe (Tage)
LOOKBACK_MAP = {
    '5m':  60,   '15m': 60,
    '30m': 365,  '1h':  365,
    '2h':  730,  '4h':  730,
    '6h':  1095, '1d':  1095, '1w': 1095,
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str):
    os.makedirs(LOG_DIR, exist_ok=True)
    line = f"{datetime.now().isoformat()} AUTO-OPTIMIZER-WBOT {msg}"
    with open(TRIGGER_LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _format_elapsed(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def _get_last_run() -> datetime | None:
    if not os.path.exists(LAST_RUN_FILE):
        return None
    with open(LAST_RUN_FILE, 'r') as f:
        s = f.read().strip()
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _set_last_run():
    os.makedirs(CACHE_DIR, exist_ok=True)
    now_str = datetime.now().isoformat()
    with open(LAST_RUN_FILE, 'w') as f:
        f.write(now_str)
    _log(f"LAST_RUN updated={now_str}")


def _is_due(schedule: dict) -> tuple[bool, str]:
    if os.path.exists(IN_PROGRESS_FILE):
        _log("SKIP already_in_progress")
        return False, None

    last_run = _get_last_run()
    if last_run is None:
        return True, 'first_run'

    interval_cfg     = schedule.get('interval', {})
    value            = int(interval_cfg.get('value', 7))
    unit             = interval_cfg.get('unit', 'days')
    multipliers      = {'minutes': 60, 'hours': 3600, 'days': 86400, 'weeks': 604800}
    interval_seconds = value * multipliers.get(unit, 86400)

    if (datetime.now() - last_run).total_seconds() >= interval_seconds:
        return True, 'interval'

    now    = datetime.now()
    dow    = int(schedule.get('day_of_week', 0))
    hour   = int(schedule.get('hour', 3))
    minute = int(schedule.get('minute', 0))
    if now.weekday() == dow and now.hour == hour and minute <= now.minute < minute + 15:
        if last_run.date() < now.date():
            return True, 'scheduled'

    return False, None


# ---------------------------------------------------------------------------
# Paare / Symbole / Timeframes aufloesen
# ---------------------------------------------------------------------------

def _resolve_pairs_auto(live_settings: dict) -> list:
    """[(full_symbol, timeframe)] aus aktiven Strategien."""
    pairs, seen = [], set()
    for s in live_settings.get('active_strategies', []):
        if not s.get('active', True):
            continue
        sym = s.get('symbol', '')
        tf  = s.get('timeframe', '')
        if sym and tf and (sym, tf) not in seen:
            pairs.append((sym, tf))
            seen.add((sym, tf))
    return pairs or [('BTC/USDT:USDT', '1d'), ('ETH/USDT:USDT', '4h')]


def _resolve_timeframes(value, live_settings: dict) -> list:
    if value != 'auto':
        return value if isinstance(value, list) else [value]
    seen, tfs = set(), []
    for _, tf in _resolve_pairs_auto(live_settings):
        if tf not in seen:
            tfs.append(tf)
            seen.add(tf)
    return tfs or ['1d', '4h']


def _resolve_lookback(value, timeframes: list) -> int:
    if value != 'auto':
        return int(value)
    return max((LOOKBACK_MAP.get(tf, 365) for tf in timeframes), default=365)


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def _get_telegram_credentials():
    try:
        with open(SECRET_FILE, 'r') as f:
            secrets = json.load(f)
        tg = secrets.get('telegram', {})
        return tg.get('bot_token'), tg.get('chat_id')
    except Exception:
        return None, None


def _send_telegram_plain(message: str):
    bot_token, chat_id = _get_telegram_credentials()
    if not bot_token or not chat_id:
        _log("TELEGRAM SKIP kein token/chat_id in secret.json")
        return
    try:
        import requests
        api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        requests.post(api_url, data={'chat_id': chat_id, 'text': message}, timeout=10)
        _log("TELEGRAM sent")
    except Exception as e:
        _log(f"TELEGRAM ERROR {e}")


def _send_start_telegram(pair_display: list, num_trials: int, start_time: datetime):
    msg = (
        f"\U0001f30a wbot QGRS Auto-Optimizer GESTARTET\n"
        f"Paare: {', '.join(pair_display)}\n"
        f"Trials: {num_trials}\n"
        f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    _send_telegram_plain(msg)


def _send_end_telegram(elapsed_seconds: float):
    dur = _format_elapsed(elapsed_seconds)

    if not os.path.exists(OPTIMIZER_RESULTS_FILE):
        _send_telegram_plain(f"\u2705 wbot QGRS Auto-Optimizer abgeschlossen\nDauer: {dur}")
        return

    try:
        with open(OPTIMIZER_RESULTS_FILE, encoding='utf-8') as f:
            results = json.load(f)
    except Exception:
        _send_telegram_plain(f"\u2705 wbot QGRS Auto-Optimizer abgeschlossen (Dauer: {dur})")
        return

    saved  = results.get('saved',  [])
    failed = results.get('failed', [])
    total  = len(saved) + len(failed)

    lines = [f"\u2705 wbot QGRS Auto-Optimizer abgeschlossen (Dauer: {dur})"]

    if saved:
        lines.append(f"\n\u2714 Gespeichert ({len(saved)}/{total}):")
        for s in saved:
            sym_short = s['symbol'].split('/')[0]
            lines.append(
                f"\u2022 {sym_short}/{s['timeframe']}: +{s['pnl_pct']}% \u2192 {s['config_file']}"
            )

    if failed:
        lines.append(f"\n\u274c Fehlgeschlagen ({len(failed)}/{total}):")
        for fi in failed:
            sym_short = fi['symbol'].split('/')[0]
            lines.append(f"\u2022 {sym_short}/{fi['timeframe']}: {fi['reason']}")

    _send_telegram_plain('\n'.join(lines))


# ---------------------------------------------------------------------------
# Pipeline-Ausfuehrung
# ---------------------------------------------------------------------------

def _run_python_pairs(pairs: list, lookback: int, opt_settings: dict) -> int:
    """Fuehrt den Optimizer direkt als Python-Prozess fuer explizite Paare aus."""
    python_exe  = sys.executable
    start_date  = (date.today() - timedelta(days=lookback)).strftime('%Y-%m-%d')
    end_date    = date.today().strftime('%Y-%m-%d')
    pairs_str   = ' '.join(f"{sym}|{tf}" for sym, tf in pairs)
    constraints = opt_settings.get('constraints', {})

    display = [f"{s.split('/')[0]}/{t}" for s, t in pairs]
    _log(f"PIPELINE_EXEC pairs={display}")

    cmd = [
        python_exe, OPTIMIZER_SCRIPT,
        '--pairs',         pairs_str,
        '--start_date',    start_date,
        '--end_date',      end_date,
        '--jobs',          str(opt_settings.get('cpu_cores', -1)),
        '--max_drawdown',  str(constraints.get('max_drawdown_pct', 30)),
        '--start_capital', str(opt_settings.get('start_capital', 1000)),
        '--min_win_rate',  str(constraints.get('min_win_rate_pct', 40)),
        '--trials',        str(opt_settings.get('num_trials', 200)),
        '--min_pnl',       str(constraints.get('min_pnl_pct', 0)),
        '--mode',          opt_settings.get('mode', 'strict'),
    ]

    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(PROJECT_ROOT, 'src')

    result = subprocess.run(cmd, env=env)
    rc = result.returncode
    _log(f"PIPELINE_EXIT rc={rc}")
    return rc


# ---------------------------------------------------------------------------
# Haupt-Ablauf
# ---------------------------------------------------------------------------

def run_optimization(
    schedule: dict,
    opt_settings: dict,
    live_settings: dict,
    reason: str
):
    os.makedirs(CACHE_DIR, exist_ok=True)

    pairs        = _resolve_pairs_auto(live_settings)
    pair_display = [f"{sym.split('/')[0]}/{tf}" for sym, tf in pairs]
    timeframes   = list(dict.fromkeys(tf for _, tf in pairs))

    lookback   = _resolve_lookback(opt_settings.get('lookback_days', 'auto'), timeframes)
    start_time = datetime.now()
    num_trials = int(opt_settings.get('num_trials', 200))

    _log(f"START reason={reason} pairs={pair_display} lookback_days={lookback} trials={num_trials}")

    with open(IN_PROGRESS_FILE, 'w') as f:
        f.write(start_time.isoformat())

    send_tg = opt_settings.get('send_telegram_on_completion', False)
    if send_tg:
        _send_start_telegram(pair_display, num_trials, start_time)

    start_perf = time.time()
    success    = False

    try:
        rc      = _run_python_pairs(pairs, lookback, opt_settings)
        success = (rc == 0)
    except Exception as e:
        _log(f"ERROR {e}")
    finally:
        if os.path.exists(IN_PROGRESS_FILE):
            os.remove(IN_PROGRESS_FILE)

    elapsed = round(time.time() - start_perf, 1)

    if success:
        _set_last_run()
        _log(f"FINISH result=success elapsed_s={elapsed}")
        if send_tg:
            _send_end_telegram(elapsed)
    else:
        _log(f"FINISH result=failed elapsed_s={elapsed}")


# ---------------------------------------------------------------------------
# Einstiegspunkt
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='wbot QGRS Auto-Optimizer Scheduler')
    parser.add_argument('--force', action='store_true',
                        help='Optimierung sofort erzwingen (ignoriert Zeitplan)')
    args = parser.parse_args()

    try:
        with open(SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
    except Exception as e:
        print(f"Fehler beim Lesen der settings.json: {e}")
        return

    opt_settings  = settings.get('optimization_settings', {})
    live_settings = settings.get('live_trading_settings', {})

    if not opt_settings.get('enabled', False) and not args.force:
        print("Auto-Optimierung deaktiviert (optimization_settings.enabled=false).")
        return

    schedule = opt_settings.get('schedule', {
        'day_of_week': 6, 'hour': 15, 'minute': 0,
        'interval': {'value': 7, 'unit': 'days'},
    })

    if args.force:
        reason = 'forced'
    else:
        due, reason = _is_due(schedule)
        if not due:
            print("Optimierung noch nicht faellig.")
            return

    run_optimization(schedule, opt_settings, live_settings, reason)


if __name__ == '__main__':
    main()
