# src/wbot/analysis/backtester.py
# Backtest-Engine fuer wbot QGRS (Multi-Symbol/Timeframe, Config-Dateien)
import os
import sys
import logging
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

logger = logging.getLogger(__name__)

FEE_PCT       = 0.0006   # 0.06% Taker-Fee
SLIPPAGE_PCT  = 0.0005   # 0.05% Slippage
MIN_HISTORY   = 200      # Mindest-Kerzen fuer Physik-Berechnungen


def load_data(symbol: str, timeframe: str, start_date_str: str, end_date_str: str) -> pd.DataFrame:
    """
    Laedt OHLCV-Daten fuer einen Zeitraum (mit Cache).
    Re-Export aus data_fetcher.

    Args:
        symbol: Handelspaar (z.B. "BTC/USDT:USDT")
        timeframe: Zeitrahmen
        start_date_str: "YYYY-MM-DD"
        end_date_str:   "YYYY-MM-DD"

    Returns:
        DataFrame mit [open, high, low, close, volume]
    """
    from wbot.utils.data_fetcher import load_data as _load
    return _load(symbol, timeframe, start_date_str, end_date_str)


def run_backtest(
    df: pd.DataFrame,
    config: dict,
    start_capital: float = 1000.0,
    verbose: bool = False,
    n_sim: int = 500,
) -> dict:
    """
    Fuehrt einen vollstaendigen Backtest ueber historische Kerzen durch.

    Fuer jede Kerze (ab Index 200): MarketState → MonteCarlo → Forecast → Signal → Trade.

    Config-Format:
        config = {
          "market":   {"symbol": "BTC/USDT:USDT", "timeframe": "1d"},
          "physics":  {"garch_window": 100, "fractal_window": 100,
                       "chaos_window": 50, "info_window": 30, "liquidity_bins": 100},
          "strategy": {"breakout_threshold_pct": 1.5, "breakout_prob_threshold": 0.6,
                       "range_threshold_pct": 0.5, "range_prob_threshold": 0.6},
          "risk":     {"risk_per_trade_pct": 2.0, "leverage": 10}
        }

    Args:
        df: OHLCV-DataFrame (mindestens MIN_HISTORY + 10 Kerzen)
        config: vollstaendiges Config-Dict (market/physics/strategy/risk)
        start_capital: Startkapital in USDT
        verbose: Fortschritts-Ausgabe
        n_sim: Anzahl Monte-Carlo-Simulationen pro Kerze

    Returns:
        dict mit Metriken: total_pnl_pct, trades_count, win_rate,
                           max_drawdown_pct, end_capital, equity_curve (DataFrame)
    """
    from wbot.model.market_state import compute_market_state
    from wbot.model.monte_carlo import run_monte_carlo
    from wbot.forecast.range_forecast import compute_range_forecast
    from wbot.strategy.signal_logic import generate_signal

    # Parameter aus Config extrahieren
    strategy_cfg = config.get('strategy', {})
    risk_cfg     = config.get('risk', {})
    leverage     = risk_cfg.get('leverage', 10)
    risk_per_trade_pct = risk_cfg.get('risk_per_trade_pct', 2.0)

    # signal_logic liest risk-Parameter aus strategy_cfg, also fuegen wir sie dort ein
    merged_strategy_cfg = {**strategy_cfg, **risk_cfg}

    if len(df) < MIN_HISTORY + 10:
        logger.warning(
            f"Zu wenig Daten fuer Backtest: {len(df)} Kerzen (min {MIN_HISTORY + 10} benoetigt)."
        )
        return _empty_result(start_capital)

    capital = start_capital
    position = None
    trades = []
    equity_curve = []

    start_idx = MIN_HISTORY

    for i in range(start_idx, len(df)):
        current_row   = df.iloc[i]
        current_price = float(current_row['close'])
        candle_high   = float(current_row['high'])
        candle_low    = float(current_row['low'])
        ts            = df.index[i]

        # --- Offene Position pruefen (SL/TP) ---
        if position is not None:
            pos_side    = position['side']
            sl_price    = position['sl_price']
            tp_price    = position['tp_price']
            entry_price = position['entry_price']
            notional    = position['notional']

            hit_sl = (
                (pos_side == 'long'  and candle_low  <= sl_price) or
                (pos_side == 'short' and candle_high >= sl_price)
            )
            hit_tp = (
                (pos_side == 'long'  and candle_high >= tp_price) or
                (pos_side == 'short' and candle_low  <= tp_price)
            )

            if hit_sl or hit_tp:
                exit_price = sl_price if hit_sl else tp_price
                if pos_side == 'long':
                    pnl_pct = (exit_price / entry_price - 1.0)
                else:
                    pnl_pct = (1.0 - exit_price / entry_price)

                pnl_usd = notional * pnl_pct
                fee     = notional * FEE_PCT * 2
                pnl_usd -= fee

                capital += pnl_usd
                result   = 'win' if pnl_usd > 0 else 'loss'

                trades.append({
                    'entry_time':   position['entry_time'],
                    'exit_time':    ts,
                    'side':         pos_side,
                    'entry_price':  entry_price,
                    'exit_price':   exit_price,
                    'pnl_usdt':     pnl_usd,
                    'result':       result,
                    'capital_after': max(0.0, capital),
                    'exit_reason':  'sl' if hit_sl else 'tp',
                })
                position = None

                if capital <= 0:
                    equity_curve.append({'timestamp': ts, 'equity': 0.0})
                    break

        # Equity-Tracking (ohne offene Position vereinfacht)
        equity_curve.append({'timestamp': ts, 'equity': capital})

        # --- Neuen Entry nur wenn keine offene Position ---
        if position is not None:
            continue

        if verbose and (i - start_idx) % 50 == 0:
            pct = (i - start_idx) / max(1, len(df) - start_idx) * 100
            print(f"  Backtest: {pct:.0f}% | Kapital: {capital:.2f} | Trades: {len(trades)}")

        # MarketState berechnen
        try:
            window_df = df.iloc[:i].copy()
            state = compute_market_state(window_df, config=config)
        except Exception as e:
            logger.debug(f"MarketState-Fehler Kerze {i}: {e}")
            continue

        # Monte-Carlo
        try:
            sim_result = run_monte_carlo(state, n_simulations=n_sim, n_steps=50)
        except Exception as e:
            logger.debug(f"Monte-Carlo-Fehler Kerze {i}: {e}")
            continue

        # Forecast
        try:
            forecast = compute_range_forecast(sim_result, current_price)
        except Exception as e:
            logger.debug(f"Forecast-Fehler Kerze {i}: {e}")
            continue

        # Signal (merged_strategy_cfg enthaelt sowohl strategy- als auch risk-Parameter)
        try:
            signal = generate_signal(forecast, state, current_price, merged_strategy_cfg)
        except Exception as e:
            logger.debug(f"Signal-Fehler Kerze {i}: {e}")
            continue

        if signal.action == 'wait':
            continue

        # Trade-Entry berechnen
        side = 'long' if 'long' in signal.action else 'short'
        stop_distance_pct = abs(signal.entry_price - signal.stop_loss) / max(signal.entry_price, 1e-10)

        if stop_distance_pct < 0.0001:
            continue

        risk_amount = capital * (risk_per_trade_pct / 100.0)
        # Notional = Risiko / Stop-Distance
        notional    = risk_amount / stop_distance_pct

        # Hebel-Limit: nicht mehr als Kapital * Hebel
        max_notional = capital * leverage
        notional = min(notional, max_notional)

        if notional < 1.0:
            continue

        # Entry mit Slippage
        if side == 'long':
            entry_price = signal.entry_price * (1 + SLIPPAGE_PCT)
        else:
            entry_price = signal.entry_price * (1 - SLIPPAGE_PCT)

        position = {
            'side':        side,
            'entry_price': entry_price,
            'sl_price':    signal.stop_loss,
            'tp_price':    signal.take_profit,
            'notional':    notional,
            'entry_time':  ts,
        }

    # Offene Position am Ende schliessen
    if position is not None and len(df) > 0:
        last_price  = float(df['close'].iloc[-1])
        pos_side    = position['side']
        entry_price = position['entry_price']
        notional    = position['notional']

        if pos_side == 'long':
            pnl_pct = (last_price / entry_price - 1.0)
        else:
            pnl_pct = (1.0 - last_price / entry_price)

        pnl_usd = notional * pnl_pct - notional * FEE_PCT * 2
        capital += pnl_usd

        trades.append({
            'entry_time':    position['entry_time'],
            'exit_time':     df.index[-1],
            'side':          pos_side,
            'entry_price':   entry_price,
            'exit_price':    last_price,
            'pnl_usdt':      pnl_usd,
            'result':        'win' if pnl_usd > 0 else 'loss',
            'capital_after': max(0.0, capital),
            'exit_reason':   'end_of_data',
        })

    return _compute_metrics(trades, start_capital, capital, equity_curve)


def _empty_result(start_capital: float) -> dict:
    """Gibt ein leeres Ergebnis-Dict zurueck."""
    equity_df = pd.DataFrame(columns=['timestamp', 'equity'])
    return {
        'total_pnl_pct':     0.0,
        'trades_count':      0,
        'win_rate':          0.0,
        'max_drawdown_pct':  0.0,
        'end_capital':       start_capital,
        'equity_curve':      equity_df,
    }


def _compute_metrics(
    trades: list,
    start_capital: float,
    final_capital: float,
    equity_curve: list
) -> dict:
    """Berechnet Performance-Metriken."""
    final_capital = max(0.0, final_capital)
    total_pnl_pct = (final_capital / start_capital - 1.0) * 100.0 if start_capital > 0 else 0.0

    equity_df = pd.DataFrame(equity_curve) if equity_curve else pd.DataFrame(columns=['timestamp', 'equity'])

    if not equity_df.empty and 'equity' in equity_df.columns:
        equity_df['equity'] = equity_df['equity'].clip(lower=0)
        peak = equity_df['equity'].cummax()
        dd   = (peak - equity_df['equity']) / peak.replace(0, np.nan)
        max_drawdown_pct = float(dd.max(skipna=True) * 100.0) if len(dd) > 0 else 0.0
        if np.isnan(max_drawdown_pct):
            max_drawdown_pct = 0.0
    else:
        max_drawdown_pct = 0.0

    if not trades:
        return {
            'total_pnl_pct':    total_pnl_pct,
            'trades_count':     0,
            'win_rate':         0.0,
            'max_drawdown_pct': max_drawdown_pct,
            'end_capital':      final_capital,
            'equity_curve':     equity_df,
        }

    wins     = sum(1 for t in trades if t['result'] == 'win')
    win_rate = wins / len(trades) * 100.0

    return {
        'total_pnl_pct':    round(total_pnl_pct, 4),
        'trades_count':     len(trades),
        'win_rate':         round(win_rate, 2),
        'max_drawdown_pct': round(max_drawdown_pct, 4),
        'end_capital':      round(final_capital, 4),
        'equity_curve':     equity_df,
    }
