# src/wbot/analysis/backtester.py
# Backtest-Engine fuer wbot QGRS
import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

logger = logging.getLogger(__name__)

FEE_PCT = 0.0006       # 0.06% Taker-Fee
SLIPPAGE_PCT = 0.0005  # 0.05% Slippage

# Backtest nutzt weniger Simulationen fuer Geschwindigkeit
BACKTEST_N_SIMULATIONS = 2000
BACKTEST_N_STEPS = 100       # Vereinfachter Intraday-Pfad
BACKTEST_WINDOW = 100        # Fenster fuer MarketState-Berechnung


def run_backtest(
    df: pd.DataFrame,
    config: dict,
    start_capital: float = 1000.0,
    verbose: bool = True,
) -> dict:
    """
    Fuehrt einen vollstaendigen Backtest ueber historische Tageskerzen durch.

    Fuer jede Kerze (ab Window-Offset): MarketState → Simulation → Signal → Trade-Simulation.

    Args:
        df: OHLCV-DataFrame (historisch, mindestens BACKTEST_WINDOW + 10 Kerzen)
        config: settings.json dict
        start_capital: Startkapital in USDT
        verbose: Fortschritts-Logging

    Returns:
        Dict mit Performance-Metriken
    """
    from wbot.model.market_state import compute_market_state
    from wbot.model.monte_carlo import run_monte_carlo
    from wbot.forecast.range_forecast import compute_range_forecast
    from wbot.strategy.signal_logic import generate_signal

    strategy_cfg = config.get('strategy', {})
    leverage = strategy_cfg.get('leverage', 10)

    if len(df) < BACKTEST_WINDOW + 10:
        logger.error(
            f"Zu wenig Daten fuer Backtest: {len(df)} Kerzen. "
            f"Minimum: {BACKTEST_WINDOW + 10}"
        )
        return {'error': 'insufficient_data'}

    capital = start_capital
    position = None
    trades = []

    # Iteriere ab BACKTEST_WINDOW-ter Kerze
    start_idx = BACKTEST_WINDOW
    n_total = len(df) - start_idx

    logger.info(f"Starte Backtest: {n_total} Kerzen | Startkapital: {start_capital:.2f} USDT")
    logger.info(f"Simulationen pro Kerze: {BACKTEST_N_SIMULATIONS} | Schritte: {BACKTEST_N_STEPS}")

    for i in range(start_idx, len(df)):
        # Verwende letzte BACKTEST_WINDOW Kerzen als Context
        window_df = df.iloc[max(0, i - BACKTEST_WINDOW):i].copy()
        current_row = df.iloc[i]
        current_price = float(current_row['close'])
        candle_high = float(current_row['high'])
        candle_low = float(current_row['low'])

        progress = i - start_idx + 1
        if verbose and progress % 20 == 0:
            logger.info(
                f"Backtest Fortschritt: {progress}/{n_total} | "
                f"Kapital: {capital:.2f} USDT | Trades: {len(trades)}"
            )

        # --- Offene Position pruefen (SL/TP) ---
        if position is not None:
            pos_side = position['side']
            sl_price = position['sl_price']
            tp_price = position['tp_price']
            entry_price = position['entry_price']
            amount = position['amount']
            notional = position['notional']

            hit_sl = (
                (pos_side == 'long' and candle_low <= sl_price) or
                (pos_side == 'short' and candle_high >= sl_price)
            )
            hit_tp = (
                (pos_side == 'long' and candle_high >= tp_price) or
                (pos_side == 'short' and candle_low <= tp_price)
            )

            if hit_sl or hit_tp:
                exit_price = sl_price if hit_sl else tp_price
                # Slippage
                if pos_side == 'long':
                    exit_price = exit_price * (1 - SLIPPAGE_PCT)
                    pnl = (exit_price - entry_price) * amount
                else:
                    exit_price = exit_price * (1 + SLIPPAGE_PCT)
                    pnl = (entry_price - exit_price) * amount

                # Fees (Entry + Exit, auf Notional-Basis)
                fee = notional * (FEE_PCT * 2)
                pnl -= fee

                capital += pnl
                result = 'win' if pnl > 0 else 'loss'

                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': df.index[i],
                    'side': pos_side,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_usdt': pnl,
                    'result': result,
                    'capital_after': capital,
                    'exit_reason': 'sl' if hit_sl else 'tp',
                })
                position = None

        # --- Neuen Entry nur wenn keine offene Position ---
        if position is not None:
            continue

        # MarketState berechnen (mit try/except fuer Robustheit)
        try:
            state = compute_market_state(window_df, config=config)
        except Exception as e:
            logger.debug(f"MarketState-Fehler bei Kerze {i}: {e}")
            continue

        # Monte-Carlo (weniger Simulationen fuer Backtest-Geschwindigkeit)
        try:
            sim_result = run_monte_carlo(
                state,
                n_simulations=BACKTEST_N_SIMULATIONS,
                n_steps=BACKTEST_N_STEPS
            )
        except Exception as e:
            logger.debug(f"Monte-Carlo-Fehler bei Kerze {i}: {e}")
            continue

        # Forecast
        try:
            forecast = compute_range_forecast(sim_result, current_price)
        except Exception as e:
            logger.debug(f"Forecast-Fehler bei Kerze {i}: {e}")
            continue

        # Signal
        try:
            signal = generate_signal(forecast, state, current_price, strategy_cfg)
        except Exception as e:
            logger.debug(f"Signal-Fehler bei Kerze {i}: {e}")
            continue

        if signal.action == 'wait':
            continue

        # --- Trade-Entry simulieren ---
        side = 'long' if 'long' in signal.action else 'short'

        # Positionsgroesse: nutze risk_per_trade_pct vom aktuellen Kapital
        risk_per_trade_pct = strategy_cfg.get('risk_per_trade_pct', 2.0)
        risk_amount = capital * (risk_per_trade_pct / 100.0)
        stop_distance_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price

        if stop_distance_pct < 0.0001:
            continue

        margin_needed = risk_amount / stop_distance_pct
        notional = margin_needed * leverage
        amount = notional / current_price  # Asset-Einheiten

        # Entry mit Slippage
        if side == 'long':
            entry_price = current_price * (1 + SLIPPAGE_PCT)
        else:
            entry_price = current_price * (1 - SLIPPAGE_PCT)

        position = {
            'side': side,
            'entry_price': entry_price,
            'sl_price': signal.stop_loss,
            'tp_price': signal.take_profit,
            'amount': amount,
            'notional': notional,
            'entry_time': df.index[i],
            'signal_action': signal.action,
        }

        logger.debug(
            f"Trade Entry: {side} @ {entry_price:.4f} | "
            f"SL={signal.stop_loss:.4f} | TP={signal.take_profit:.4f}"
        )

    # Noch offene Position am Ende schliessen (zum Schlusskurs)
    if position is not None:
        last_price = float(df['close'].iloc[-1])
        pos_side = position['side']
        entry_price = position['entry_price']
        amount = position['amount']
        notional = position['notional']

        if pos_side == 'long':
            pnl = (last_price - entry_price) * amount
        else:
            pnl = (entry_price - last_price) * amount

        fee = notional * (FEE_PCT * 2)
        pnl -= fee
        capital += pnl
        result = 'win' if pnl > 0 else 'loss'

        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': df.index[-1],
            'side': pos_side,
            'entry_price': entry_price,
            'exit_price': last_price,
            'pnl_usdt': pnl,
            'result': result,
            'capital_after': capital,
            'exit_reason': 'end_of_data',
        })

    return _compute_metrics(trades, start_capital, capital)


def _compute_metrics(trades: list, start_capital: float, final_capital: float) -> dict:
    """Berechnet Performance-Metriken aus der Trade-Liste."""
    if not trades:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'pnl_usdt': 0.0,
            'pnl_pct': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0,
            'calmar_ratio': 0.0,
            'avg_win_usdt': 0.0,
            'avg_loss_usdt': 0.0,
            'start_capital': start_capital,
            'final_capital': start_capital,
        }

    df_trades = pd.DataFrame(trades)
    wins = df_trades[df_trades['result'] == 'win']
    losses = df_trades[df_trades['result'] == 'loss']

    total_pnl = final_capital - start_capital
    pnl_pct = total_pnl / start_capital * 100.0

    # Max Drawdown
    capital_series = pd.Series([start_capital] + list(df_trades['capital_after']))
    rolling_max = capital_series.cummax()
    drawdown = (capital_series - rolling_max) / rolling_max * 100.0
    max_drawdown_pct = float(abs(drawdown.min()))

    # Calmar Ratio
    calmar = (pnl_pct / max_drawdown_pct) if max_drawdown_pct > 0 else 0.0

    # Sharpe-Ratio (vereinfacht, aus Trade-PnLs)
    pnl_series = df_trades['pnl_usdt'].values
    if len(pnl_series) > 1 and np.std(pnl_series) > 0:
        sharpe = float(np.mean(pnl_series) / np.std(pnl_series) * np.sqrt(len(pnl_series)))
    else:
        sharpe = 0.0

    return {
        'total_trades': len(trades),
        'winning_trades': int(len(wins)),
        'losing_trades': int(len(losses)),
        'win_rate': float(len(wins) / len(trades) * 100.0) if trades else 0.0,
        'pnl_usdt': float(total_pnl),
        'pnl_pct': float(pnl_pct),
        'max_drawdown_pct': max_drawdown_pct,
        'sharpe_ratio': sharpe,
        'calmar_ratio': float(calmar),
        'avg_win_usdt': float(wins['pnl_usdt'].mean()) if len(wins) > 0 else 0.0,
        'avg_loss_usdt': float(losses['pnl_usdt'].mean()) if len(losses) > 0 else 0.0,
        'start_capital': start_capital,
        'final_capital': float(final_capital),
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="wbot QGRS Backtester")
    parser.add_argument('--symbol', required=True, type=str, help="Handelspaar (z.B. BTC/USDT:USDT)")
    parser.add_argument('--timeframe', default='1d', type=str, help="Zeitrahmen (default: 1d)")
    parser.add_argument('--start-capital', type=float, default=1000.0, help="Startkapital USDT")
    args = parser.parse_args()

    symbol = args.symbol
    timeframe = args.timeframe

    # Settings laden
    settings_path = os.path.join(PROJECT_ROOT, 'settings.json')
    try:
        with open(settings_path) as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Fehler beim Laden von settings.json: {e}")
        config = {}

    # Daten laden
    from wbot.utils.data_fetcher import fetch_ohlcv
    logger.info(f"Lade historische Daten fuer {symbol} ({timeframe})...")
    df = fetch_ohlcv(symbol, timeframe, limit=500)

    if df is None or len(df) < BACKTEST_WINDOW + 10:
        logger.error(
            f"Nicht genug Daten fuer Backtest. "
            f"Benoetigt: {BACKTEST_WINDOW + 10}, Erhalten: {len(df) if df is not None else 0}"
        )
        sys.exit(1)

    logger.info(f"Backtest-Daten: {len(df)} Kerzen | {df.index[0]} bis {df.index[-1]}")

    # Backtest ausfuehren
    metrics = run_backtest(df, config, start_capital=args.start_capital)

    if 'error' in metrics:
        logger.error(f"Backtest-Fehler: {metrics['error']}")
        sys.exit(1)

    # Ergebnisse speichern
    results_dir = os.path.join(PROJECT_ROOT, 'artifacts', 'results')
    os.makedirs(results_dir, exist_ok=True)
    safe_name = f"{symbol.replace('/', '').replace(':', '')}_{timeframe}"
    results_path = os.path.join(results_dir, f"backtest_{safe_name}.json")

    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Ergebnisse gespeichert: {results_path}")

    # Report ausgeben
    from wbot.analysis.show_results import show_backtest_results
    show_backtest_results(metrics)


if __name__ == "__main__":
    main()
