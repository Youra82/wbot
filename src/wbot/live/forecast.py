# src/wbot/live/forecast.py
# Tagesvorhersage: alle Stationen aktualisieren, Modelle wie im Walk-Forward (Stand Monatsbeginn) trainieren,
# Wahrscheinlichkeiten fuer heute berechnen und fuer die Vorwaertspruefung protokollieren.
import os
import json
import logging
import numpy as np
import pandas as pd

from wbot.world import sources as src
from wbot.world.features import STATIONS, atr14
from wbot.forecast.ensemble import build_panel, fit_month, combine, decide

logger = logging.getLogger(__name__)
PROJECT_ROOT = src.PROJECT_ROOT
FORECAST_LOG = os.path.join(PROJECT_ROOT, 'artifacts', 'live', 'forecasts.csv')
RESEARCH = os.path.join(PROJECT_ROOT, 'artifacts', 'results', 'research.json')


def research_result():
    if not os.path.exists(RESEARCH):
        return {}
    with open(RESEARCH, encoding='utf-8') as f:
        return json.load(f)


def ensemble_stations(live):
    st = live.get('ensemble_stations', 'auto')
    if st == 'auto':
        st = research_result().get('aufgenommen') or STATIONS
    unknown = [s for s in st if s not in STATIONS]
    if unknown:
        raise ValueError(f"Unbekannte Stationen in settings.json: {unknown}")
    return list(st)


def today_utc():
    return pd.Timestamp.utcnow().tz_localize(None).normalize()


def forecast_today(live, today=None):
    """Rueckgabe: (panel, Tabelle je Coin mit Station-/Ensemble-Wahrscheinlichkeiten und Entscheidung)."""
    today = today or today_utc()
    failed = src.refresh_all()
    coins = live.get('coins', src.COINS)
    panel = build_panel(coins)
    rows = panel[panel.index == today]
    missing = sorted(set(coins) - set(rows['coin']))
    if missing:
        raise RuntimeError(f"Keine Merkmalszeile fuer {today.date()} bei {missing} (Kursdaten noch nicht da?)")
    models, clim = fit_month(panel, today.replace(day=1), STATIONS)
    preds = {st: m.predict(rows) for st, m in models.items()}
    ens = ensemble_stations(live)
    pL, pS = combine({s: preds[s] for s in ens}, clim)
    table = pd.DataFrame({'coin': rows['coin'].values, 'climL': clim[0], 'climS': clim[1]}, index=rows.index)
    for st in STATIONS:
        table[f'{st}_pL'], table[f'{st}_pS'] = preds[st]
    table['ens_pL'], table['ens_pS'] = pL, pS
    table['side'] = decide(pL, pS)
    table['atr'] = [float(atr14(src.binance_klines(c)).iloc[-1]) for c in table['coin']]
    table.attrs['failed_sources'] = failed
    table.attrs['ensemble'] = ens
    _log(table)
    return panel, table


def _log(table):
    os.makedirs(os.path.dirname(FORECAST_LOG), exist_ok=True)
    t = table.drop(columns=['atr']).copy()
    t.index.name = 'date'
    if os.path.exists(FORECAST_LOG):
        old = pd.read_csv(FORECAST_LOG, index_col=0, parse_dates=True)
        old = old[~((old.index.isin(t.index)) & (old['coin'].isin(t['coin'])))]
        t = pd.concat([old, t])
    t.sort_index().to_csv(FORECAST_LOG)


def forward_verification(panel):
    """BSS je Station und Ensemble auf allen live protokollierten, inzwischen aufgeloesten Vorhersagen."""
    if not os.path.exists(FORECAST_LOG):
        return {}, 0
    fc = pd.read_csv(FORECAST_LOG, index_col=0, parse_dates=True)
    lab = panel[['coin', 'WL', 'WS']].dropna()
    m = fc.reset_index().merge(lab.reset_index(), on=['date', 'coin'])
    if m.empty:
        return {}, 0
    out = {}
    for st in STATIONS + ['ens']:
        b = []
        for side, y, cl in (('pL', 'WL', 'climL'), ('pS', 'WS', 'climS')):
            p = m[f'{st}_{side}'].values
            ok = np.isfinite(p)
            if ok.sum() == 0:
                continue
            b.append(1 - np.mean((p[ok] - m[y].values[ok]) ** 2) / np.mean((m[cl].values[ok] - m[y].values[ok]) ** 2))
        out[st] = float(np.mean(b)) if b else np.nan
    return out, len(m)
