# src/wbot/analysis/portfolio_simulator.py
# Portfolio-Simulation fuer wbot QGRS (mehrere Strategien auf gemeinsamer Kapital-Basis)
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from wbot.analysis.backtester import run_backtest

logger = logging.getLogger(__name__)

SETTINGS_FILE = os.path.join(PROJECT_ROOT, 'settings.json')


def _load_max_open_positions() -> int:
    """Laedt max_open_positions aus settings.json (Default: 5)."""
    try:
        with open(SETTINGS_FILE) as f:
            s = json.load(f)
        return int(s.get('live_trading_settings', {}).get('max_open_positions', 5))
    except Exception:
        return 5


def run_portfolio_simulation(
    start_capital: float,
    strategies_data: dict,
    start_date: str,
    end_date: str
) -> dict:
    """
    Fuehrt eine chronologische Portfolio-Simulation durch.

    strategies_data Format:
        {
          "config_BTCUSDTUSDT_1d.json": {
            "symbol":    "BTC/USDT:USDT",
            "timeframe": "1d",
            "data":      pd.DataFrame,   # OHLCV
            "config":    dict,            # vollstaendige Config
          },
          ...
        }

    Args:
        start_capital:    Startkapital in USDT
        strategies_data:  Dict mit Strategien-Daten
        start_date:       "YYYY-MM-DD" (wird nicht direkt genutzt, Daten bereits geladen)
        end_date:         "YYYY-MM-DD" (wird nicht direkt genutzt)

    Returns:
        dict mit start_capital, end_capital, total_pnl_pct, trade_count,
             win_rate, max_drawdown_pct, max_drawdown_date, liquidation_date,
             equity_curve (pd.DataFrame)
    """
    print("\n--- Starte wbot QGRS Portfolio-Simulation... ---")

    max_open_positions = _load_max_open_positions()

    # --- 1. Backtest fuer jede Strategie einzeln ausfuehren ---
    print("1/3: Berechne Signale fuer alle Strategien...")

    strategy_results = {}
    for key, strat in tqdm(strategies_data.items(), desc="Berechne Strategien"):
        df     = strat.get('data')
        config = strat.get('config', {})
        if df is None or df.empty or len(df) < 210:
            logger.warning(f"Ueberspringe {key}: zu wenig Daten.")
            continue
        try:
            bt = run_backtest(df, config, start_capital=start_capital, verbose=False, n_sim=200)
            strategy_results[key] = {
                'data':      df,
                'config':    config,
                'bt_result': bt,
                'symbol':    strat.get('symbol', ''),
                'timeframe': strat.get('timeframe', ''),
            }
        except Exception as e:
            logger.warning(f"Backtest-Fehler bei {key}: {e}")
            continue

    if not strategy_results:
        print("Keine gueltigen Strategien nach Backtest.")
        return None

    # --- 2. Zeitachse zusammenstellen (Union aller Timestamps) ---
    print("2/3: Erstelle gemeinsame Zeitachse...")
    all_timestamps = set()
    for key, sr in strategy_results.items():
        all_timestamps.update(sr['data'].index)
    sorted_timestamps = sorted(list(all_timestamps))
    print(f"   -> {len(sorted_timestamps)} Zeitschritte zu simulieren.")

    # --- 3. Chronologische Simulation ---
    print("3/3: Fuehre Portfolio-Simulation durch...")

    equity          = start_capital
    peak_equity     = start_capital
    max_drawdown    = 0.0
    max_dd_date     = None
    liquidation_dt  = None
    open_positions  = {}   # key -> position-dict
    trade_history   = []
    equity_curve    = []

    fee_pct      = 0.0006
    slippage_pct = 0.0005

    # Pre-compute signals fuer jede Strategie (aus equity_curve des Backtests)
    # Einfacherer Ansatz: Wir simulieren Trades tagesweise basierend auf
    # den equity_curve-Daten des Backtests (Trade-by-Trade-Replay)

    # Baue Trade-Listen aus Backtest-Ergebnissen auf
    # Der Backtester liefert keine Trade-Liste direkt, daher simulieren wir
    # pro Zeitstempel mit den Signalen neu (vereinfacht: nutze equity_curve als Proxy)

    # Vereinfachte Portfolio-Simulation:
    # Pro Zeitstempel: prueffe ob eine Strategie ein neues Signal gibt
    # (basierend auf der Equity-Kurve des Backtests)

    # Besserer Ansatz: Baue Equity-Delta pro Zeitschritt
    # equity_delta[ts] = sum aller offenen Positionen PnL in diesem Step

    # Wir rekonstrieren: aus dem Backtest equity_curve berechnen wir
    # den tagesweisen Return jeder Strategie und kombinieren sie

    strategy_equity_series = {}
    for key, sr in strategy_results.items():
        eq_curve = sr['bt_result'].get('equity_curve')
        if eq_curve is None or eq_curve.empty:
            continue
        # Erstelle eine Zeitreihe mit dem Equity je Timestamp
        eq_curve = eq_curve.copy()
        if 'timestamp' in eq_curve.columns:
            eq_curve = eq_curve.set_index('timestamp')
        eq_curve.index = pd.to_datetime(eq_curve.index, utc=True, errors='coerce')
        strategy_equity_series[key] = eq_curve['equity']

    if not strategy_equity_series:
        print("Keine Equity-Kurven verfuegbar. Breche ab.")
        return None

    # Kombiniere alle Equity-Kurven auf gemeinsamer Zeitachse
    combined_eq = pd.DataFrame(strategy_equity_series)
    combined_eq = combined_eq.sort_index()
    combined_eq = combined_eq.ffill()

    # Berechne Tages-Returns je Strategie
    returns_df = combined_eq.pct_change().fillna(0.0)

    # Portfolio-Return = Durchschnitt der Returns aller aktiven Strategien
    # (gleichgewichtet, max max_open_positions gleichzeitig)
    equity     = start_capital
    equity_pts = []

    for ts in tqdm(combined_eq.index, desc="Portfolio-Simulation"):
        if liquidation_dt:
            break

        # Aktive Strategien an diesem Timestamp (max max_open_positions)
        row_returns = returns_df.loc[ts].dropna()
        active_cols = [c for c in row_returns.index if c in strategy_results]

        # Begrenze auf max_open_positions Strategien (Top-N nach letzter Performance)
        if len(active_cols) > max_open_positions:
            active_cols = active_cols[:max_open_positions]

        if active_cols:
            avg_ret = row_returns[active_cols].mean()
            equity  = equity * (1 + avg_ret)

        equity_pts.append({'timestamp': ts, 'equity': equity})

        # Drawdown tracken
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if dd > max_drawdown:
            max_drawdown = dd
            max_dd_date  = ts

        if equity <= 0:
            liquidation_dt = ts
            break

    # Trade-Statistiken aus Backtest-Ergebnissen aggregieren
    total_trades = sum(sr['bt_result'].get('trades_count', 0) for sr in strategy_results.values())
    weighted_wr  = 0.0
    if total_trades > 0:
        total_wins = sum(
            int(sr['bt_result'].get('trades_count', 0) * sr['bt_result'].get('win_rate', 0) / 100)
            for sr in strategy_results.values()
        )
        weighted_wr = total_wins / total_trades * 100.0

    equity_df = pd.DataFrame(equity_pts)
    if not equity_df.empty:
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        equity_df['peak']      = equity_df['equity'].cummax()
        dd_col = (equity_df['peak'] - equity_df['equity']) / equity_df['peak'].replace(0, np.nan)
        equity_df['drawdown_pct'] = dd_col.fillna(0.0)
        equity_df.set_index('timestamp', inplace=True, drop=False)

    final_equity  = equity_pts[-1]['equity'] if equity_pts else start_capital
    total_pnl_pct = (final_equity / start_capital - 1.0) * 100.0 if start_capital > 0 else 0.0

    return {
        "start_capital":     start_capital,
        "end_capital":       round(final_equity, 4),
        "total_pnl_pct":     round(total_pnl_pct, 4),
        "trade_count":       total_trades,
        "win_rate":          round(weighted_wr, 2),
        "max_drawdown_pct":  round(max_drawdown * 100.0, 4),
        "max_drawdown_date": max_dd_date,
        "liquidation_date":  liquidation_dt,
        "equity_curve":      equity_df,
    }
