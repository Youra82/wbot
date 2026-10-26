# src/wbot/analysis/show_results.py
# Zeigt: praeregistriertes Testergebnis, Papierkonto, letzte Vorhersagen.
import os
import sys
import json
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
RES = os.path.join(PROJECT_ROOT, 'artifacts', 'results', 'research.json')
PAPER = os.path.join(PROJECT_ROOT, 'artifacts', 'live', 'paper_state.json')
FC = os.path.join(PROJECT_ROOT, 'artifacts', 'live', 'forecasts.csv')


def main():
    if os.path.exists(RES):
        r = json.load(open(RES, encoding='utf-8'))
        print(f"=== Praeregistrierter Test (erstellt {r.get('erstellt', '?')[:16]}) ===")
        for st, v in r['validierung'].items():
            print(f"  {st:15s} BSS L={v['bss_long']:+.4f} S={v['bss_short']:+.4f} Trades {v['trades']:4d} "
                  f"avgR={v['avgR'] if v['avgR'] == v['avgR'] else float('nan'):+.3f} -> "
                  f"{'aufgenommen' if v['aufgenommen'] else 'raus'}")
        e = r.get('endtest_ensemble', {})
        print(f"  Endtest Ensemble {r['aufgenommen']}: {json.dumps(e, default=str)}")
        print(f"  ==> {'BESTANDEN' if r['bestanden'] else 'NICHT BESTANDEN'}\n")
    else:
        print('Noch kein Testergebnis -- ./run_pipeline.sh ausfuehren.\n')
    if os.path.exists(PAPER):
        p = json.load(open(PAPER, encoding='utf-8'))
        print(f"=== Papierkonto: {p['equity']:.2f} USDT (Start {p['start_capital']:.0f}) ===")
        if p['closed']:
            c = pd.DataFrame(p['closed'])
            print(f"  {len(c)} Trades, Ø {c['R'].mean():+.3f} R, Trefferquote {(c['R'] > 0).mean() * 100:.0f} %")
            print(c.tail(10)[['coin', 'side', 'date', 'exit_date', 'reason', 'R']].to_string(index=False))
        for coin, pos in p['open'].items():
            print(f"  offen: {coin} {pos['side']} seit {pos['date']}")
        print()
    if os.path.exists(FC):
        f = pd.read_csv(FC, index_col=0, parse_dates=True)
        last = f[f.index == f.index.max()]
        print(f"=== Letzte Vorhersage {f.index.max().date()} ({f.index.nunique()} Tage protokolliert) ===")
        print(last[['coin', 'ens_pL', 'ens_pS', 'side']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
