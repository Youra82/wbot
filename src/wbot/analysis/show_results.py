# src/wbot/analysis/show_results.py
# Ergebnis-Analyse fuer wbot QGRS (4 Modi, analog zu stbot)
import os
import sys
import json
import argparse
import logging
import warnings
import pandas as pd
from datetime import date

warnings.filterwarnings('ignore')
logging.getLogger('optuna').setLevel(logging.ERROR)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from wbot.analysis.backtester import load_data, run_backtest
from wbot.analysis.portfolio_simulator import run_portfolio_simulation
from wbot.analysis.portfolio_optimizer import run_portfolio_optimizer

logger = logging.getLogger(__name__)


def _send_telegram_document(csv_path: str, caption: str):
    """Sendet einen CSV-Bericht via Telegram (optional)."""
    try:
        secret_path = os.path.join(PROJECT_ROOT, 'secret.json')
        with open(secret_path, 'r') as f:
            secrets = json.load(f)
        tg = secrets.get('telegram', {})
        token   = tg.get('bot_token')
        chat_id = tg.get('chat_id')
        if not token or not chat_id:
            return

        import requests
        with open(csv_path, 'rb') as doc:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendDocument",
                data={'chat_id': chat_id, 'caption': caption},
                files={'document': doc},
                timeout=15
            )
        print("Bericht wurde an Telegram gesendet.")
    except Exception as e:
        print(f"Telegram-Versand fehlgeschlagen: {e}")


def _save_equity_csv(equity_df: pd.DataFrame, path: str):
    """Speichert Equity-Kurve als CSV."""
    try:
        export_cols = ['timestamp', 'equity', 'drawdown_pct']
        available   = [c for c in export_cols if c in equity_df.columns]
        equity_df[available].to_csv(path, index=False)
        print(f"Equity-Kurve exportiert: {os.path.basename(path)}")
    except Exception as e:
        print(f"CSV-Export-Fehler: {e}")


def _plot_equity_ascii(equity_df: pd.DataFrame, title: str, width: int = 60, height: int = 12):
    """Gibt eine einfache ASCII-Art Equity-Kurve aus."""
    if equity_df is None or equity_df.empty or 'equity' not in equity_df.columns:
        print(f"  [Keine Equity-Daten fuer '{title}']")
        return

    values = equity_df['equity'].values
    if len(values) < 2:
        return

    min_v = values.min()
    max_v = values.max()
    if max_v == min_v:
        print(f"  {title}: Equity konstant bei {min_v:.2f} USDT")
        return

    # Normiere auf 0..height
    def normalize(v):
        return int((v - min_v) / (max_v - min_v) * (height - 1))

    # Resample auf 'width' Punkte
    step     = max(1, len(values) // width)
    sampled  = values[::step][:width]
    n        = len(sampled)

    canvas = [[' '] * n for _ in range(height)]
    for x, v in enumerate(sampled):
        y = height - 1 - normalize(v)
        y = max(0, min(height - 1, y))
        canvas[y][x] = '*'

    print(f"\n  {title}")
    print(f"  {max_v:>8.0f} |" + "─" * n)
    for row in canvas:
        print(f"           |" + ''.join(row))
    print(f"  {min_v:>8.0f} |" + "─" * n)
    print(f"  {'Start':>8}   {'Ende':>{n - 8}}")
    print()


# ─── Modus 1: Einzel-Analyse ────────────────────────────────────────────────

def run_single_analysis(start_date: str, end_date: str, start_capital: float):
    """Analysiert jede Config-Datei isoliert."""
    print("--- wbot QGRS Ergebnis-Analyse (Einzel-Modus) ---")
    configs_dir = os.path.join(PROJECT_ROOT, 'src', 'wbot', 'strategy', 'configs')

    if not os.path.isdir(configs_dir):
        print(f"Konfigurationsverzeichnis nicht gefunden: {configs_dir}")
        return

    config_files = sorted([
        f for f in os.listdir(configs_dir)
        if f.startswith('config_') and f.endswith('.json')
    ])

    if not config_files:
        print("Keine Konfigurationsdateien gefunden.")
        return

    print(f"Zeitraum: {start_date} bis {end_date} | Startkapital: {start_capital:.2f} USDT")
    all_results = []

    for filename in config_files:
        config_path = os.path.join(configs_dir, filename)
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception as e:
            print(f"Config-Lesefehler {filename}: {e}")
            continue

        symbol    = config.get('market', {}).get('symbol', '')
        timeframe = config.get('market', {}).get('timeframe', '')
        if not symbol or not timeframe:
            continue

        strategy_name = f"{symbol} ({timeframe})"
        print(f"\nAnalysiere: {filename}...")

        data = load_data(symbol, timeframe, start_date, end_date)
        if data is None or data.empty:
            print(f"  WARNUNG: Keine Daten fuer {strategy_name}. Ueberspringe.")
            continue

        try:
            result = run_backtest(data, config, start_capital=start_capital, verbose=False, n_sim=500)
        except Exception as e:
            print(f"  FEHLER bei {filename}: {e}")
            continue

        all_results.append({
            "Strategie":   strategy_name,
            "Trades":      result.get('trades_count', 0),
            "Win Rate %":  result.get('win_rate', 0.0),
            "PnL %":       result.get('total_pnl_pct', 0.0),
            "Max DD %":    result.get('max_drawdown_pct', 0.0),
            "Endkapital":  result.get('end_capital', start_capital),
        })

    if not all_results:
        print("Keine gueltigen Ergebnisse.")
        return

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="PnL %", ascending=False)

    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', '{:.2f}'.format)

    sep = "=" * 90
    print(f"\n\n{sep}")
    print(f"              wbot QGRS - Zusammenfassung aller Einzelstrategien")
    print(sep)
    print(results_df.to_string(index=False))
    print(sep)


# ─── Modus 2 + 3: Geteilter Modus (Manuell / Auto) ──────────────────────────

def run_shared_mode(
    is_auto: bool,
    start_date: str,
    end_date: str,
    start_capital: float,
    target_max_dd: float
):
    """Portfolio-Simulation (manuell) oder Portfolio-Optimierung (auto)."""
    mode_name = "Automatische Portfolio-Optimierung" if is_auto else "Manuelle Portfolio-Simulation"
    print(f"--- wbot QGRS {mode_name} ---")
    if is_auto:
        print(f"Ziel: Maximaler Profit bei maximal {target_max_dd:.2f}% Drawdown.")

    configs_dir = os.path.join(PROJECT_ROOT, 'src', 'wbot', 'strategy', 'configs')
    available_strategies = []
    if os.path.isdir(configs_dir):
        for filename in sorted(os.listdir(configs_dir)):
            if filename.startswith('config_') and filename.endswith('.json'):
                available_strategies.append(filename)

    if not available_strategies:
        print("Keine optimierten Config-Dateien gefunden.")
        return

    # Auswahl
    selected_files = []
    if not is_auto:
        print("\nVerfuegbare Strategien:")
        for i, name in enumerate(available_strategies):
            print(f"  {i+1}) {name}")
        selection = input("\nWelche Strategien simulieren? (Zahlen mit Komma, z.B. 1,3 oder 'alle'): ")
        try:
            if selection.lower() == 'alle':
                selected_files = available_strategies
            else:
                selected_files = [
                    available_strategies[int(x.strip()) - 1]
                    for x in selection.split(',')
                ]
        except (ValueError, IndexError):
            print("Ungueltige Auswahl. Breche ab.")
            return
    else:
        selected_files = available_strategies

    # Daten laden
    strategies_data = {}
    print("\nLade Daten fuer ausgewaehlte Strategien...")
    for filename in selected_files:
        try:
            with open(os.path.join(configs_dir, filename)) as f:
                config = json.load(f)
            symbol    = config['market']['symbol']
            timeframe = config['market']['timeframe']
            data      = load_data(symbol, timeframe, start_date, end_date)
            if data is not None and not data.empty:
                strategies_data[filename] = {
                    'symbol':    symbol,
                    'timeframe': timeframe,
                    'data':      data,
                    'config':    config,
                }
            else:
                print(f"  WARNUNG: Keine Daten fuer {filename}. Wird ignoriert.")
        except Exception as e:
            print(f"  FEHLER bei {filename}: {e}")

    if not strategies_data:
        print("Konnte fuer keine Strategie Daten laden. Breche ab.")
        return

    equity_df = pd.DataFrame()
    csv_path  = ""
    caption   = ""

    try:
        if is_auto:
            results = run_portfolio_optimizer(
                start_capital, strategies_data, start_date, end_date, target_max_dd
            )

            if results and results.get('final_result'):
                final_report = results['final_result']
                sep = "=" * 60
                print(f"\n{sep}")
                print("     wbot QGRS - Automatische Portfolio-Optimierung")
                print(sep)
                print(f"Zeitraum: {start_date} bis {end_date}")
                print(f"Startkapital: {start_capital:.2f} USDT")
                print(f"Bedingung: Max Drawdown <= {target_max_dd:.2f}%")

                portfolio = results.get('optimal_portfolio', [])
                print(f"\nOptimales Portfolio ({len(portfolio)} Strategien):")
                for fn in portfolio:
                    print(f"  - {fn}")

                print("\n--- Portfolio Performance ---")
                print(f"Endkapital:      {final_report['end_capital']:.2f} USDT")
                print(f"Gesamt PnL:      {final_report['end_capital'] - start_capital:+.2f} USDT "
                      f"({final_report['total_pnl_pct']:.2f}%)")
                print(f"Portfolio Max DD: {final_report['max_drawdown_pct']:.2f}%")
                liq = final_report.get('liquidation_date')
                print(f"Liquidiert:      {'JA, am ' + str(liq)[:10] if liq else 'NEIN'}")
                print(sep)

                csv_path  = os.path.join(PROJECT_ROOT, 'optimal_portfolio_equity.csv')
                caption   = (f"wbot QGRS Portfolio-Optimierung (Max DD <= {target_max_dd:.1f}%)\n"
                             f"Endkapital: {final_report['end_capital']:.2f} USDT")
                equity_df = final_report.get('equity_curve', pd.DataFrame())
            else:
                print(f"\nKein Portfolio gefunden, das Max DD <= {target_max_dd:.2f}% erfuellt.")

        else:
            # Manuelle Simulation
            sim_data = {
                f"{v['symbol']}_{v['timeframe']}": v
                for k, v in strategies_data.items()
            }
            results = run_portfolio_simulation(start_capital, sim_data, start_date, end_date)

            if results:
                sep = "=" * 60
                print(f"\n{sep}")
                print("           wbot QGRS Portfolio-Simulations-Ergebnis")
                print(sep)
                print(f"Zeitraum: {start_date} bis {end_date}")
                print(f"Startkapital: {results['start_capital']:.2f} USDT")
                print("\n--- Gesamt-Performance ---")
                print(f"Endkapital:      {results['end_capital']:.2f} USDT")
                print(f"Gesamt PnL:      {results['end_capital'] - results['start_capital']:+.2f} USDT "
                      f"({results['total_pnl_pct']:.2f}%)")
                print(f"Anzahl Trades:   {results['trade_count']}")
                print(f"Win-Rate:        {results['win_rate']:.2f}%")
                dd_date = results.get('max_drawdown_date')
                print(f"Portfolio Max DD: {results['max_drawdown_pct']:.2f}% "
                      f"am {str(dd_date)[:10] if dd_date else 'N/A'}")
                liq = results.get('liquidation_date')
                print(f"Liquidiert:      {'JA, am ' + str(liq)[:10] if liq else 'NEIN'}")
                print(sep)

                csv_path  = os.path.join(PROJECT_ROOT, 'manual_portfolio_equity.csv')
                caption   = (f"wbot QGRS Portfolio-Simulation\n"
                             f"Endkapital: {results['end_capital']:.2f} USDT")
                equity_df = results.get('equity_curve', pd.DataFrame())

    except Exception as e:
        import traceback
        print(f"\nFEHLER waehrend Analyse: {e}")
        traceback.print_exc()
        equity_df = pd.DataFrame()

    # Export
    if isinstance(equity_df, pd.DataFrame) and not equity_df.empty and csv_path:
        print("\n--- Export ---")
        _save_equity_csv(equity_df, csv_path)
        _send_telegram_document(csv_path, caption)


# ─── Modus 4: Interaktive Charts ────────────────────────────────────────────

def run_interactive_charts(start_date: str, end_date: str, start_capital: float):
    """Equity-Kurven aller Configs als ASCII-Art oder matplotlib PNG."""
    print("--- wbot QGRS Interaktive Charts ---")
    configs_dir = os.path.join(PROJECT_ROOT, 'src', 'wbot', 'strategy', 'configs')

    if not os.path.isdir(configs_dir):
        print(f"Konfigurationsverzeichnis nicht gefunden: {configs_dir}")
        return

    config_files = sorted([
        f for f in os.listdir(configs_dir)
        if f.startswith('config_') and f.endswith('.json')
    ])

    if not config_files:
        print("Keine Konfigurationsdateien gefunden.")
        return

    charts_dir = os.path.join(PROJECT_ROOT, 'artifacts', 'charts')
    os.makedirs(charts_dir, exist_ok=True)

    for filename in config_files:
        config_path = os.path.join(configs_dir, filename)
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception:
            continue

        symbol    = config.get('market', {}).get('symbol', '')
        timeframe = config.get('market', {}).get('timeframe', '')
        if not symbol or not timeframe:
            continue

        data = load_data(symbol, timeframe, start_date, end_date)
        if data is None or data.empty:
            print(f"Keine Daten fuer {filename}. Ueberspringe.")
            continue

        try:
            result    = run_backtest(data, config, start_capital=start_capital, verbose=False, n_sim=500)
            equity_df = result.get('equity_curve')
        except Exception as e:
            print(f"Backtest-Fehler bei {filename}: {e}")
            continue

        title = f"{symbol} ({timeframe}) | PnL: {result.get('total_pnl_pct', 0):.2f}% | " \
                f"WR: {result.get('win_rate', 0):.1f}% | MaxDD: {result.get('max_drawdown_pct', 0):.1f}%"

        # ASCII-Chart
        _plot_equity_ascii(equity_df, title)

        # Optionale matplotlib PNG
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            if isinstance(equity_df, pd.DataFrame) and not equity_df.empty and 'equity' in equity_df.columns:
                fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

                axes[0].plot(equity_df.index if isinstance(equity_df.index, pd.DatetimeIndex)
                             else range(len(equity_df)),
                             equity_df['equity'].values, color='steelblue', linewidth=1.5)
                axes[0].axhline(y=start_capital, color='gray', linestyle='--', linewidth=0.8)
                axes[0].set_title(title, fontsize=10)
                axes[0].set_ylabel('Kapital (USDT)')
                axes[0].grid(True, alpha=0.3)

                if 'drawdown_pct' in equity_df.columns:
                    axes[1].fill_between(
                        equity_df.index if isinstance(equity_df.index, pd.DatetimeIndex)
                        else range(len(equity_df)),
                        equity_df['drawdown_pct'].values * 100,
                        0, color='red', alpha=0.4
                    )
                    axes[1].set_ylabel('Drawdown (%)')
                    axes[1].set_xlabel('Zeit')
                    axes[1].grid(True, alpha=0.3)

                plt.tight_layout()

                safe_name  = f"{symbol.replace('/', '').replace(':', '')}_{timeframe}"
                chart_path = os.path.join(charts_dir, f"equity_{safe_name}.png")
                plt.savefig(chart_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  Chart gespeichert: {chart_path}")

        except ImportError:
            pass  # matplotlib nicht verfuegbar, ASCII reicht
        except Exception as e:
            logger.debug(f"Matplotlib-Fehler: {e}")


# ─── Hauptprogramm ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="wbot QGRS Ergebnis-Analyse")
    parser.add_argument('--mode', default='1', type=str, choices=['1', '2', '3', '4'],
                        help="1=Einzel, 2=Manuell, 3=Auto, 4=Charts")
    parser.add_argument('--target_max_drawdown', default=30.0, type=float,
                        help="Ziel Max Drawdown %% (nur fuer Modus 3)")
    args = parser.parse_args()

    print("\n--- Bitte Zeitraum und Kapital festlegen ---")
    start_date    = input("Startdatum (JJJJ-MM-TT) [Standard: 2023-01-01]: ").strip() or "2023-01-01"
    end_date      = input("Enddatum   (JJJJ-MM-TT) [Standard: Heute]: ").strip() or date.today().strftime("%Y-%m-%d")
    try:
        start_capital = float(input("Startkapital in USDT [Standard: 1000]: ").strip() or "1000")
    except ValueError:
        start_capital = 1000.0
    print("---------------------------------------------------")

    if args.mode == '1':
        run_single_analysis(start_date, end_date, start_capital)
    elif args.mode == '2':
        run_shared_mode(False, start_date, end_date, start_capital, target_max_dd=999.0)
    elif args.mode == '3':
        run_shared_mode(True, start_date, end_date, start_capital, target_max_dd=args.target_max_drawdown)
    elif args.mode == '4':
        run_interactive_charts(start_date, end_date, start_capital)
