# src/wbot/live/report.py
# Telegram-Wetterbericht: Wahrscheinlichkeiten je Coin, staerkste Station, Papierkonto, Vorwaertspruefung.
import numpy as np

from wbot.world.features import STATIONS

NAMES = {'preis': 'Preis', 'makro': 'Makro', 'stablecoins': 'Stablecoins', 'coinbase': 'Coinbase',
         'korea': 'Korea', 'aufmerksamkeit': 'Wikipedia', 'derivate': 'Funding', 'ens': 'Ensemble'}


def _pct(x):
    return '–' if not np.isfinite(x) else f'{x * 100:.0f}%'


def weather_report(today, table, paper_state, closed, opened, live_msgs, fwd, n_fwd, mode):
    lines = [f'🌦 <b>wbot Weltwetter {today:%d.%m.%Y}</b> ({mode})',
             f'Ensemble: {", ".join(NAMES[s] for s in table.attrs.get("ensemble", []))}',
             f'Klima: Long-Ziel {_pct(table["climL"].iloc[0])} | Short-Ziel {_pct(table["climS"].iloc[0])}', '']
    for r in table.itertuples():
        # Station mit der groessten Abweichung vom Klima
        dev = {st: max(abs(getattr(r, f'{st}_pL') - r.climL), abs(getattr(r, f'{st}_pS') - r.climS))
               for st in STATIONS if np.isfinite(getattr(r, f'{st}_pL'))}
        top = max(dev, key=dev.get) if dev else None
        act = {'long': '🟢 LONG', 'short': '🔴 SHORT'}.get(r.side, '⚪ kein Trade')
        lines.append(f'<b>{r.coin}</b> L {_pct(r.ens_pL)} / S {_pct(r.ens_pS)} → {act}'
                     + (f' (stärkste Station: {NAMES[top]})' if top else ''))
    failed = table.attrs.get('failed_sources') or []
    if failed:
        lines += ['', f'⚠️ Quellen nicht aktualisiert: {", ".join(failed)}']
    lines += ['', f'📒 Papierkonto: {paper_state["equity"]:.2f} USDT (Start {paper_state["start_capital"]:.0f}), '
                  f'{len(paper_state["closed"])} Trades abgeschlossen']
    for t in closed:
        lines.append(f'  ✔ {t["coin"]} {t["side"]} {t["reason"]} {t["R"]:+.2f}R')
    for t in opened:
        lines.append(f'  ➕ {t["coin"]} {t["side"]} eröffnet')
    if paper_state['open']:
        lines.append('  offen: ' + ', '.join(f'{c} {p["side"]} seit {p["date"]}' for c, p in paper_state['open'].items()))
    if live_msgs:
        lines += ['', '💰 Echtgeld:'] + [f'  {m}' for m in live_msgs]
    if n_fwd:
        lines += ['', f'🔬 Vorwärtsprüfung ({n_fwd} aufgelöste Vorhersagen, BSS > 0 = besser als Klima):']
        lines.append('  ' + ' | '.join(f'{NAMES[s]} {v:+.3f}' for s, v in fwd.items() if np.isfinite(v)))
    return '\n'.join(lines)
