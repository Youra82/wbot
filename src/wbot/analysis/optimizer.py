# src/wbot/analysis/optimizer.py
# Optuna-Optimierer fuer wbot QGRS (Multi-Symbol/Timeframe)
import os
import sys
import json
import optuna
import argparse
import logging
import warnings
from datetime import datetime as _dt

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from wbot.analysis.backtester import load_data, run_backtest

logger = logging.getLogger(__name__)

# Globale Variablen fuer Optuna Objective (werden vor jedem Lauf gesetzt)
HISTORICAL_DATA          = None
CURRENT_SYMBOL           = None
CURRENT_TIMEFRAME        = None
MAX_DRAWDOWN_CONSTRAINT  = 0.30
MIN_WIN_RATE_CONSTRAINT  = 40.0
MIN_PNL_CONSTRAINT       = 0.0
START_CAPITAL            = 1000.0
OPTIM_MODE               = "strict"

RESULTS_FILE = os.path.join(PROJECT_ROOT, 'artifacts', 'results', 'last_optimizer_run.json')


def create_safe_filename(symbol: str, timeframe: str) -> str:
    return f"{symbol.replace('/', '').replace(':', '')}_{timeframe}"


def objective(trial: optuna.Trial) -> float:
    """Optuna Objective-Funktion: Sucht optimale QGRS-Parameter."""
    physics_params = {
        'garch_window':   trial.suggest_int('garch_window',   50, 200),
        'fractal_window': trial.suggest_int('fractal_window', 50, 200),
        'chaos_window':   trial.suggest_int('chaos_window',   30, 100),
        'info_window':    trial.suggest_int('info_window',    10,  50),
        'liquidity_bins': trial.suggest_int('liquidity_bins', 50, 200),
    }
    strategy_params = {
        'breakout_threshold_pct':  trial.suggest_float('breakout_threshold_pct',  0.5, 3.0),
        'breakout_prob_threshold': trial.suggest_float('breakout_prob_threshold', 0.45, 0.80),
        'range_threshold_pct':     trial.suggest_float('range_threshold_pct',     0.3, 1.5),
        'range_prob_threshold':    trial.suggest_float('range_prob_threshold',    0.45, 0.80),
    }
    risk_params = {
        'risk_per_trade_pct': trial.suggest_float('risk_per_trade_pct', 0.5, 5.0),
        'leverage':           trial.suggest_int('leverage', 3, 20),
    }

    config = {
        'market':   {'symbol': CURRENT_SYMBOL, 'timeframe': CURRENT_TIMEFRAME},
        'physics':  physics_params,
        'strategy': strategy_params,
        'risk':     risk_params,
    }

    try:
        result = run_backtest(
            HISTORICAL_DATA.copy(),
            config,
            start_capital=START_CAPITAL,
            verbose=False,
            n_sim=300,
        )
    except Exception as e:
        logger.debug(f"run_backtest Exception: {e}")
        raise optuna.exceptions.TrialPruned()

    pnl_pct      = result.get('total_pnl_pct', -9999.0)
    drawdown_pct = result.get('max_drawdown_pct', 100.0)
    win_rate     = result.get('win_rate', 0.0)
    trades_count = result.get('trades_count', 0)

    if OPTIM_MODE == "strict":
        if (
            drawdown_pct > MAX_DRAWDOWN_CONSTRAINT
            or win_rate < MIN_WIN_RATE_CONSTRAINT
            or pnl_pct < MIN_PNL_CONSTRAINT
            or trades_count < 5
        ):
            raise optuna.exceptions.TrialPruned()
    elif OPTIM_MODE == "best_profit":
        if drawdown_pct > MAX_DRAWDOWN_CONSTRAINT or trades_count < 5:
            raise optuna.exceptions.TrialPruned()

    return pnl_pct


def main():
    global HISTORICAL_DATA, CURRENT_SYMBOL, CURRENT_TIMEFRAME
    global MAX_DRAWDOWN_CONSTRAINT, MIN_WIN_RATE_CONSTRAINT, MIN_PNL_CONSTRAINT
    global START_CAPITAL, OPTIM_MODE

    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="wbot QGRS Parameter-Optimierung")
    parser.add_argument('--symbols',       type=str, default="",
                        help='Space-getrennte Basiswaehrungen, z.B. "BTC ETH"')
    parser.add_argument('--timeframes',    type=str, default="",
                        help='Space-getrennte Timeframes, z.B. "1d 4h"')
    parser.add_argument('--pairs',         type=str, default="",
                        help='Explizite Paare: "BTC/USDT:USDT|1d ETH/USDT:USDT|4h"')
    parser.add_argument('--start_date',    required=True, type=str)
    parser.add_argument('--end_date',      required=True, type=str)
    parser.add_argument('--jobs',          required=True, type=int)
    parser.add_argument('--max_drawdown',  required=True, type=float)
    parser.add_argument('--start_capital', required=True, type=float)
    parser.add_argument('--min_win_rate',  required=True, type=float)
    parser.add_argument('--trials',        required=True, type=int)
    parser.add_argument('--min_pnl',       required=True, type=float)
    parser.add_argument('--mode',          required=True, type=str,
                        choices=['strict', 'best_profit'])
    args = parser.parse_args()

    MAX_DRAWDOWN_CONSTRAINT = args.max_drawdown
    MIN_WIN_RATE_CONSTRAINT = args.min_win_rate
    MIN_PNL_CONSTRAINT      = args.min_pnl
    START_CAPITAL           = args.start_capital
    N_TRIALS                = args.trials
    OPTIM_MODE              = args.mode

    # Tasks aufbauen
    if args.pairs.strip():
        TASKS = []
        for p in args.pairs.strip().split():
            sym, tf = p.rsplit('|', 1)
            TASKS.append({'symbol': sym, 'timeframe': tf})
    elif args.symbols and args.timeframes:
        symbols    = args.symbols.split()
        timeframes = args.timeframes.split()
        TASKS = [
            {'symbol': f"{s}/USDT:USDT", 'timeframe': tf}
            for s in symbols for tf in timeframes
        ]
    else:
        print("Fehler: --pairs oder --symbols + --timeframes muss angegeben werden.")
        return

    run_results = {
        'run_start': _dt.now().isoformat(timespec='seconds'),
        'run_end':   None,
        'saved':     [],
        'failed':    [],
    }

    for task in TASKS:
        symbol, timeframe = task['symbol'], task['timeframe']
        CURRENT_SYMBOL    = symbol
        CURRENT_TIMEFRAME = timeframe

        print(f"\n===== wbot QGRS Optimierung: {symbol} ({timeframe}) =====")

        HISTORICAL_DATA = load_data(symbol, timeframe, args.start_date, args.end_date)
        if HISTORICAL_DATA is None or HISTORICAL_DATA.empty:
            print(f"  Keine Daten verfuegbar fuer {symbol} ({timeframe}).")
            run_results['failed'].append({
                'symbol': symbol, 'timeframe': timeframe, 'reason': 'no_data'
            })
            continue

        print(f"  Daten geladen: {len(HISTORICAL_DATA)} Kerzen | "
              f"{HISTORICAL_DATA.index[0].strftime('%Y-%m-%d')} bis "
              f"{HISTORICAL_DATA.index[-1].strftime('%Y-%m-%d')}")

        DB_FILE     = os.path.join(PROJECT_ROOT, 'artifacts', 'db', 'optuna_studies_wbot.db')
        os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
        STORAGE_URL = f"sqlite:///{DB_FILE}?timeout=60"
        study_name  = f"qgrs_{create_safe_filename(symbol, timeframe)}_{OPTIM_MODE}"

        study = optuna.create_study(
            storage=STORAGE_URL,
            study_name=study_name,
            direction="maximize",
            load_if_exists=True
        )

        try:
            study.optimize(
                objective,
                n_trials=N_TRIALS,
                n_jobs=args.jobs,
                show_progress_bar=True
            )
        except Exception as e:
            print(f"FEHLER waehrend Optimierung: {e}")
            run_results['failed'].append({
                'symbol': symbol, 'timeframe': timeframe, 'reason': str(e)[:80]
            })
            continue

        valid_trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not valid_trials:
            print(f"  Keine gueltigen Trials fuer {symbol} ({timeframe}).")
            run_results['failed'].append({
                'symbol': symbol, 'timeframe': timeframe, 'reason': 'no_valid_trials'
            })
            continue

        best_trial  = max(valid_trials, key=lambda t: t.value)
        best_params = best_trial.params
        new_pnl     = best_trial.value

        print(f"\n  Bester Trial: PnL={new_pnl:.2f}%")

        # Config-Verzeichnis
        config_dir      = os.path.join(PROJECT_ROOT, 'src', 'wbot', 'strategy', 'configs')
        os.makedirs(config_dir, exist_ok=True)
        config_filename = f"config_{create_safe_filename(symbol, timeframe)}.json"
        config_path     = os.path.join(config_dir, config_filename)

        # Nur speichern wenn besser als bestehende Config
        existing_pnl = None
        if os.path.exists(config_path):
            try:
                with open(config_path) as cf:
                    existing_cfg = json.load(cf)
                existing_pnl = existing_cfg.get('_meta', {}).get('pnl_pct')
            except Exception:
                pass

        if existing_pnl is not None and new_pnl <= existing_pnl:
            print(f"  Bestehende Config besser ({existing_pnl:.2f}% vs {new_pnl:.2f}%) "
                  f"— wird nicht ueberschrieben.")
            run_results['failed'].append({
                'symbol': symbol, 'timeframe': timeframe,
                'reason': f'existing_better_{existing_pnl:.2f}pct',
            })
            continue

        config_output = {
            "market": {
                "symbol":    symbol,
                "timeframe": timeframe,
            },
            "physics": {
                "garch_window":   best_params['garch_window'],
                "fractal_window": best_params['fractal_window'],
                "chaos_window":   best_params['chaos_window'],
                "info_window":    best_params['info_window'],
                "liquidity_bins": best_params['liquidity_bins'],
            },
            "strategy": {
                "breakout_threshold_pct":  round(best_params['breakout_threshold_pct'],  4),
                "breakout_prob_threshold": round(best_params['breakout_prob_threshold'], 4),
                "range_threshold_pct":     round(best_params['range_threshold_pct'],     4),
                "range_prob_threshold":    round(best_params['range_prob_threshold'],    4),
            },
            "risk": {
                "risk_per_trade_pct": round(best_params['risk_per_trade_pct'], 4),
                "leverage":           best_params['leverage'],
            },
            "_meta": {
                "pnl_pct":      round(new_pnl, 4),
                "optimized_at": _dt.now().isoformat(timespec='seconds'),
            },
        }

        with open(config_path, 'w') as f:
            json.dump(config_output, f, indent=4)

        print(f"  [OK] Config gespeichert: {config_filename}  (PnL: {new_pnl:.2f}%)")

        run_results['saved'].append({
            'symbol':      symbol,
            'timeframe':   timeframe,
            'pnl_pct':     round(new_pnl, 2),
            'config_file': config_filename,
        })

    # Lauf-Ergebnisse speichern (fuer Scheduler / Telegram)
    run_results['run_end'] = _dt.now().isoformat(timespec='seconds')
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(run_results, f, indent=2, ensure_ascii=False)
    print(f"\nLauf-Ergebnisse gespeichert: {RESULTS_FILE}")

    saved_count  = len(run_results['saved'])
    failed_count = len(run_results['failed'])
    print(f"\n=== Optimierung abgeschlossen: {saved_count} gespeichert, {failed_count} fehlgeschlagen ===")


if __name__ == "__main__":
    main()
