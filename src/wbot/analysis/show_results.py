# src/wbot/analysis/show_results.py
# Ergebnis-Visualisierung fuer wbot QGRS
import os
import sys
import json
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

logger = logging.getLogger(__name__)


def show_backtest_results(results: dict):
    """
    Gibt einen formatierten Backtest-Report aus.

    Args:
        results: Dict mit Performance-Metriken aus run_backtest()
    """
    if 'error' in results:
        print(f"\n[FEHLER] Backtest-Fehler: {results['error']}")
        return

    win_rate = results.get('win_rate', 0)
    pnl_pct = results.get('pnl_pct', 0)
    max_dd = results.get('max_drawdown_pct', 0)

    # Bewertungs-Sterne
    stars = _rate_performance(win_rate, pnl_pct, max_dd)

    lines = [
        "",
        "=" * 60,
        "     wbot QGRS - Backtest Report",
        "=" * 60,
        "",
        f"  Bewertung:          {'*' * stars + ' ' * (5 - stars)} ({stars}/5)",
        "",
        "  TRADE-STATISTIKEN:",
        f"  Gesamt-Trades:      {results.get('total_trades', 0):>8}",
        f"  Gewinn-Trades:      {results.get('winning_trades', 0):>8}",
        f"  Verlust-Trades:     {results.get('losing_trades', 0):>8}",
        f"  Win-Rate:           {win_rate:>7.1f}%",
        "",
        "  PERFORMANCE:",
        f"  PnL:                {results.get('pnl_usdt', 0):>7.2f} USDT ({pnl_pct:.1f}%)",
        f"  Max Drawdown:       {max_dd:>7.1f}%",
        f"  Sharpe Ratio:       {results.get('sharpe_ratio', 0):>8.2f}",
        f"  Calmar Ratio:       {results.get('calmar_ratio', 0):>8.2f}",
        "",
        "  TRADE-GROESSEN:",
        f"  Avg Win:            {results.get('avg_win_usdt', 0):>7.2f} USDT",
        f"  Avg Loss:           {results.get('avg_loss_usdt', 0):>7.2f} USDT",
        "",
        "  KAPITAL:",
        f"  Startkapital:       {results.get('start_capital', 0):>7.2f} USDT",
        f"  Endkapital:         {results.get('final_capital', 0):>7.2f} USDT",
        "",
        "=" * 60,
        "",
    ]

    # Profit-Faktor (wenn vorhanden)
    avg_win = results.get('avg_win_usdt', 0)
    avg_loss = results.get('avg_loss_usdt', 0)
    if avg_loss < 0:
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        lines.insert(-2, f"  Profit-Faktor:      {profit_factor:>8.2f}")

    output = "\n".join(lines)
    print(output)
    logger.info(output)


def show_forecast(forecast, state=None):
    """
    Gibt den formatierten Range-Forecast aus.

    Args:
        forecast: RangeForecast Objekt
        state: Optionaler MarketState fuer zusaetzliche Infos
    """
    from wbot.forecast.range_forecast import format_forecast_table
    table = format_forecast_table(forecast)
    print(table)

    if state is not None:
        _show_market_state(state)


def _show_market_state(state):
    """Gibt den MarketState als formatierte Tabelle aus."""
    lines = [
        "",
        "=" * 56,
        "     wbot QGRS - MarketState (Physics-Features)",
        "=" * 56,
        "",
        "  PREIS & VOLATILITAET:",
        f"  Preis:              {state.price:>12.4f}",
        f"  GARCH-Sigma:        {state.sigma:>12.6f}",
        f"  Vol-Regime:         {state.vol_regime:>12}",
        "",
        "  FRAKTAL-ANALYSE:",
        f"  Hurst-Exponent H:   {state.hurst_H:>12.4f}",
        f"  Fraktaldimension D: {state.fractal_D:>12.4f}",
        f"  Fraktal-Regime:     {state.fractal_regime:>12}",
        "",
        "  CHAOS-INDIKATOREN:",
        f"  Lyapunov-Exponent:  {state.lyapunov:>12.4f}",
        f"  Attraktor-Regime:   {state.attractor_regime:>12}",
        "",
        "  INFORMATIONSFLUSS:",
        f"  Phi_t:              {state.info_flow:>12.6f}",
        "",
        "  LIQUIDITAETS-GRAVITATION:",
        f"  Gravitationskraft:  {state.liquidity_force:>12.6f}",
        f"  #Attraktoren:       {len(state.liquidity_attractors):>12}",
    ]

    if state.liquidity_attractors:
        lines.append("")
        lines.append("  TOP LIQUIDITAETS-ZONEN:")
        for i, (price, strength) in enumerate(state.liquidity_attractors[:5]):
            diff_pct = (price - state.price) / state.price * 100
            sign = '+' if diff_pct >= 0 else ''
            lines.append(f"    [{i+1}] {price:>12.4f}  ({sign}{diff_pct:.2f}%)  Staerke={strength:.4f}")

    lines += [
        "",
        "  MARKTENERGIE:",
        f"  E_total:            {state.energy:>12.8f}",
        "",
        "=" * 56,
        "",
    ]

    output = "\n".join(lines)
    print(output)
    logger.info(output)


def _rate_performance(win_rate: float, pnl_pct: float, max_dd_pct: float) -> int:
    """Gibt eine Bewertung von 1-5 Sternen zurueck."""
    score = 0

    if win_rate >= 55:
        score += 1
    if win_rate >= 60:
        score += 1

    if pnl_pct >= 10:
        score += 1
    if pnl_pct >= 30:
        score += 1

    if max_dd_pct <= 20:
        score += 1
    elif max_dd_pct > 50:
        score -= 1

    return max(1, min(5, score))


def _load_latest_results(symbol: str = None, timeframe: str = None) -> dict:
    """Laedt die neuesten Backtest-Ergebnisse aus dem artifacts/results-Verzeichnis."""
    results_dir = os.path.join(PROJECT_ROOT, 'artifacts', 'results')

    if not os.path.exists(results_dir):
        return {}

    result_files = [f for f in os.listdir(results_dir) if f.startswith('backtest_') and f.endswith('.json')]

    if not result_files:
        return {}

    if symbol and timeframe:
        safe_name = f"{symbol.replace('/', '').replace(':', '')}_{timeframe}"
        target = f"backtest_{safe_name}.json"
        if target in result_files:
            result_files = [target]

    # Neueste Datei laden
    result_files.sort(key=lambda f: os.path.getmtime(os.path.join(results_dir, f)), reverse=True)
    latest_file = os.path.join(results_dir, result_files[0])

    try:
        with open(latest_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Fehler beim Laden von {latest_file}: {e}")
        return {}


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    import argparse
    parser = argparse.ArgumentParser(description="wbot QGRS - Ergebnis-Anzeige")
    parser.add_argument('--symbol', type=str, help="Symbol fuer spezifische Ergebnisse")
    parser.add_argument('--timeframe', type=str, help="Timeframe fuer spezifische Ergebnisse")
    args = parser.parse_args()

    results = _load_latest_results(args.symbol, args.timeframe)

    if not results:
        print("\nKeine Backtest-Ergebnisse gefunden.")
        print("Bitte zuerst ausführen: ./run_pipeline.sh")
        return

    show_backtest_results(results)


if __name__ == "__main__":
    main()
