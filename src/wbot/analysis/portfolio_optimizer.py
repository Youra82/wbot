# src/wbot/analysis/portfolio_optimizer.py
# Greedy Portfolio-Optimierer fuer wbot QGRS
# Findet die beste Kombination von Strategien unter Max-DD-Constraint
import os
import sys
import json
import logging
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from wbot.analysis.portfolio_simulator import run_portfolio_simulation

logger = logging.getLogger(__name__)


def run_portfolio_optimizer(
    start_capital: float,
    strategies_data: dict,
    start_date: str,
    end_date: str,
    target_max_dd: float
) -> dict:
    """
    Findet die optimale Kombination von Strategien (Max-Profit unter Max-DD-Constraint).
    Verwendet Greedy-Algorithmus mit Coin-Kollisionsschutz.

    Args:
        start_capital:    Startkapital in USDT
        strategies_data:  Dict mit Strategien-Daten (Format wie portfolio_simulator)
        start_date:       Startdatum
        end_date:         Enddatum
        target_max_dd:    Maximaler zulaessiger Drawdown in % (z.B. 30.0)

    Returns:
        dict mit optimal_portfolio (Liste von Dateinamen) und final_result
    """
    print(f"\n--- wbot QGRS Portfolio-Optimierung (Max DD <= {target_max_dd:.1f}%) ---")
    target_max_dd_dec = target_max_dd / 100.0

    if not strategies_data:
        print("Keine Strategien vorhanden.")
        return {"optimal_portfolio": [], "final_result": None}

    # --- 1. Einzel-Analyse ---
    print("1/3: Analysiere Einzel-Performance und filtere nach Max DD...")
    single_results = []

    for filename, strat_data in tqdm(strategies_data.items(), desc="Einzel-Analyse"):
        if 'data' not in strat_data or strat_data['data'].empty:
            continue

        strategy_key = f"{strat_data['symbol']}_{strat_data['timeframe']}"
        sim_data     = {strategy_key: strat_data}

        try:
            result = run_portfolio_simulation(start_capital, sim_data, start_date, end_date)
        except Exception as e:
            logger.warning(f"Simulation-Fehler bei {filename}: {e}")
            continue

        if result and not result.get('liquidation_date'):
            actual_dd = result.get('max_drawdown_pct', 100.0) / 100.0
            if actual_dd <= target_max_dd_dec:
                single_results.append({'filename': filename, 'result': result})

    if not single_results:
        print(f"Keine Einzelstrategie erfuellt Max DD <= {target_max_dd:.1f}%.")
        return {"optimal_portfolio": [], "final_result": None}

    # --- 2. Bester Einzel-Startpunkt ---
    single_results.sort(key=lambda x: x['result']['end_capital'], reverse=True)

    best_portfolio_files  = [single_results[0]['filename']]
    best_portfolio_result = single_results[0]['result']
    best_end_capital      = best_portfolio_result['end_capital']
    candidate_pool        = [r['filename'] for r in single_results[1:]]

    print(f"2/3: Beste Einzelstrategie: {best_portfolio_files[0]} "
          f"(Endkapital: {best_end_capital:.2f} USDT, "
          f"Max DD: {best_portfolio_result['max_drawdown_pct']:.2f}%)")

    # Coin-Kollisionsschutz: Extrahiere verwendete Coins
    selected_coins = set()
    initial_strat  = strategies_data.get(best_portfolio_files[0])
    if initial_strat:
        coin = initial_strat['symbol'].split('/')[0]
        selected_coins.add(coin)

    # --- 3. Greedy: Fuege schrittweise die beste Erweiterung hinzu ---
    print("3/3: Suche optimale Teamerweiterung (Greedy)...")

    while True:
        best_next_file   = None
        best_next_capital = best_end_capital
        best_next_result  = best_portfolio_result

        for candidate_file in tqdm(candidate_pool, desc=f"Teste {len(best_portfolio_files)+1}-er Teams"):
            cand_data = strategies_data.get(candidate_file)
            if not cand_data:
                continue

            # Coin-Kollisionsschutz
            cand_coin = cand_data['symbol'].split('/')[0]
            if cand_coin in selected_coins:
                continue

            # Duplikat-Schutz (gleicher Symbol+Timeframe)
            unique_check = set()
            valid = True
            for f in best_portfolio_files + [candidate_file]:
                sd = strategies_data.get(f)
                if not sd:
                    valid = False
                    break
                key = sd['symbol'] + sd['timeframe']
                if key in unique_check:
                    valid = False
                    break
                unique_check.add(key)
            if not valid:
                continue

            # Team-Daten zusammenstellen
            team_files = best_portfolio_files + [candidate_file]
            team_data  = {}
            valid_data = True
            for fname in team_files:
                sd = strategies_data.get(fname)
                if sd and 'data' in sd and not sd['data'].empty:
                    sim_key          = f"{sd['symbol']}_{sd['timeframe']}"
                    team_data[sim_key] = sd
                else:
                    valid_data = False
                    break
            if not valid_data:
                continue

            # Portfolio simulieren
            try:
                result = run_portfolio_simulation(start_capital, team_data, start_date, end_date)
            except Exception as e:
                logger.debug(f"Team-Simulation-Fehler: {e}")
                continue

            if result and not result.get('liquidation_date'):
                actual_dd = result.get('max_drawdown_pct', 100.0) / 100.0
                if actual_dd <= target_max_dd_dec and result['end_capital'] > best_next_capital:
                    best_next_capital = result['end_capital']
                    best_next_file    = candidate_file
                    best_next_result  = result

        if best_next_file:
            added_strat = strategies_data.get(best_next_file)
            added_coin  = added_strat['symbol'].split('/')[0] if added_strat else ''
            selected_coins.add(added_coin)
            best_portfolio_files.append(best_next_file)
            best_end_capital      = best_next_capital
            best_portfolio_result = best_next_result
            candidate_pool.remove(best_next_file)
            print(f"-> Hinzugefuegt: {best_next_file} "
                  f"(Kapital: {best_next_capital:.2f} USDT, "
                  f"Max DD: {best_next_result['max_drawdown_pct']:.2f}%)")
        else:
            print("Keine weitere Verbesserung moeglich. Optimierung beendet.")
            break

    # Ergebnisse speichern
    try:
        results_dir = os.path.join(PROJECT_ROOT, 'artifacts', 'results')
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, 'optimization_results.json')
        save_data   = {"optimal_portfolio": best_portfolio_files}
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=4)
        print(f"Optimales Portfolio gespeichert: {output_path}")
    except Exception as e:
        logger.warning(f"Fehler beim Speichern der Optimierungsergebnisse: {e}")

    return {
        "optimal_portfolio": best_portfolio_files,
        "final_result":      best_portfolio_result,
    }
