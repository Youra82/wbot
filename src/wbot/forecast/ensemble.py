# src/wbot/forecast/ensemble.py
# "Wettermodell": je Station zwei logistische Regressionen (P Long-Ziel, P Short-Ziel), monatlich neu
# trainiert auf allen bis Monatsbeginn aufgeloesten Beobachtungen. Ensemble = Klima + Summe der
# Stations-Abweichungen in Log-Odds. Live und Backtest verwenden dieselben Funktionen.
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from wbot.world import sources as src
from wbot.world.features import coin_features, station_columns, daily_index, STATIONS
from wbot.forecast.trade_rules import label_table, P_MIN

MIN_TRAIN = 300   # Mindestzahl aufgeloester Beobachtungen je Station


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def build_panel(coins=src.COINS, stations=STATIONS, with_labels=True):
    """Merkmale (+ Ergebnisse) aller Coins in einer Tabelle, Index = (Entscheidungstag), Spalte coin."""
    parts = []
    for coin in coins:
        px = src.binance_klines(coin)
        days = daily_index(px)
        f = coin_features(coin, days, stations)
        if with_labels:
            lab = label_table(px, src.binance_funding(coin)['rate'])
            f = f.join(lab, how='left')
        f['coin'] = coin
        f.index.name = 'date'
        parts.append(f)
    return pd.concat(parts).sort_index()


class StationModel:
    def __init__(self, station, cols):
        self.station, self.cols = station, cols
        self.ok = False

    def fit(self, train):
        d = train.dropna(subset=self.cols + ['WL', 'WS'])
        if len(d) < MIN_TRAIN:
            return self
        self.scaler = StandardScaler().fit(d[self.cols].values)
        X = self.scaler.transform(d[self.cols].values)
        self.mL = LogisticRegression(C=0.1, max_iter=1000).fit(X, d['WL'].values)
        self.mS = LogisticRegression(C=0.1, max_iter=1000).fit(X, d['WS'].values)
        self.ok = True
        return self

    def predict(self, rows):
        """(pL, pS) je Zeile; NaN wenn Merkmale fehlen oder Modell nicht trainiert."""
        pL = np.full(len(rows), np.nan)
        pS = np.full(len(rows), np.nan)
        if not self.ok:
            return pL, pS
        X = rows[self.cols].values
        m = np.isfinite(X).all(axis=1)
        if m.any():
            Xs = self.scaler.transform(X[m])
            pL[m] = self.mL.predict_proba(Xs)[:, 1]
            pS[m] = self.mS.predict_proba(Xs)[:, 1]
        return pL, pS


def fit_month(panel, month_start, stations):
    """Trainiert alle Stationen mit den bis month_start aufgeloesten Beobachtungen."""
    train = panel[panel['resolved'] <= month_start]
    clim = (float(train['WL'].mean()), float(train['WS'].mean()))
    models = {st: StationModel(st, station_columns(panel.columns, st)).fit(train) for st in stations}
    return models, clim


def combine(preds, clim):
    """preds: {station: (pL, pS)} -> Ensemble (pL, pS). Fehlende Station = Klima (Beitrag 0)."""
    n = len(next(iter(preds.values()))[0]) if preds else 0
    zL = np.full(n, _logit(clim[0]))
    zS = np.full(n, _logit(clim[1]))
    for pL, pS in preds.values():
        zL += np.nan_to_num(_logit(pL) - _logit(clim[0]))
        zS += np.nan_to_num(_logit(pS) - _logit(clim[1]))
    return 1 / (1 + np.exp(-zL)), 1 / (1 + np.exp(-zS))


def decide(pL, pS, p_min=P_MIN):
    """Seite je Zeile: 'long' / 'short' / None."""
    out = np.full(len(pL), None, dtype=object)
    pLn, pSn = np.nan_to_num(pL), np.nan_to_num(pS)
    out[(pLn >= p_min) & (pLn >= pSn)] = 'long'
    out[(pSn >= p_min) & (pSn > pLn)] = 'short'
    return out


def walk_forward(panel, stations, start, end):
    """Monatliche Vorhersagen aller Stationen + Ensemble von start bis end (Entscheidungstage)."""
    out = []
    for m0 in pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq='MS'):
        m1 = m0 + pd.offsets.MonthBegin(1)
        rows = panel[(panel.index >= m0) & (panel.index < m1) & (panel.index <= pd.Timestamp(end))]
        if rows.empty:
            continue
        models, clim = fit_month(panel, m0, stations)
        res = rows[['coin', 'RL', 'RS', 'WL', 'WS', 'exitL', 'exitS']].copy()
        res['climL'], res['climS'] = clim
        preds = {}
        for st, m in models.items():
            preds[st] = m.predict(rows)
            res[f'{st}_pL'], res[f'{st}_pS'] = preds[st]
        res['ens_pL'], res['ens_pS'] = combine(preds, clim)
        out.append(res)
    return pd.concat(out)
