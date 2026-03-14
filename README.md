# wbot — QGRS (Quantum-Gravity Range System)

Probabilistisches Trading-System zur Vorhersage der **Range der nächsten Tageskerze** via Monte-Carlo-Simulation.
Inspiriert von Wettervorhersage-Ensemble-Modellen: der Bot sagt nicht "der Markt steigt", sondern
**"P(Range > 1.5%) = 63%"** — und handelt nur wenn der Erwartungswert positiv ist.

---

## Konzept

Märkte werden wie chaotische physikalische Systeme behandelt (analog zu Wetter/Turbulenz).
Der Preisvektor `X_t = (P, σ, D, λ, Φ, F_L, E)` enthält 7 Dimensionen:

| Variable | Bedeutung | Modul |
|---|---|---|
| σ | GARCH(1,1) Volatilität | `physics/garch_volatility.py` |
| D | Fraktaldimension (D = 2 - H) | `physics/fractal_dimension.py` |
| λ | Lyapunov-Exponent (Chaos-Maß) | `physics/chaos_indicators.py` |
| Φ | Informationsfluss (dI/dt) | `physics/information_flow.py` |
| F_L | Liquiditäts-Gravitationskraft | `physics/liquidity_gravity.py` |
| E | Marktenergie (multi-timeframe) | `physics/market_energy.py` |

**Pipeline:** `MarketState → GBM-Simulation × 10.000 → Range-Verteilung → Signal`

---

## Dateistruktur

```
wbot/
├── src/wbot/
│   ├── physics/           # 6 Physik-Module (Volatilität, Fraktal, Chaos, ...)
│   ├── model/             # MarketState, GBM-Preisprozess, Monte-Carlo
│   ├── forecast/          # Range-Forecast, Kerzenform, 4D-Phasenraum
│   ├── strategy/          # Signal-Logik, Live-Trading (run.py)
│   ├── analysis/          # Backtester, show_results
│   └── utils/             # data_fetcher (Bitget/ccxt), trade_manager
├── master_runner.py       # Cronjob-Orchestrator
├── settings.json          # Strategie-Konfiguration
├── secret.json            # API-Keys (nicht im Repo)
├── run_pipeline.sh        # Backtest + Ergebnisse in einem
├── show_results.sh        # Nur Ergebnisse / Live-Forecast anzeigen
├── push_configs.sh        # settings.json commiten + pushen
├── update.sh              # Sicheres Update vom Repo
└── install.sh             # Erstinstallation (venv + deps)
```

---

## Schnellstart

```bash
# 1. Installieren
./install.sh

# 2. API-Keys eintragen
cp secret.json.example secret.json
nano secret.json

# 3. Strategie konfigurieren
nano settings.json

# 4. Backtest laufen lassen
./run_pipeline.sh BTC/USDT:USDT 1d 1000

# 5. Ergebnisse anzeigen
./show_results.sh

# 6. Configs pushen (wenn live)
./push_configs.sh
```

---

## settings.json — Konfiguration

```json
{
  "live_trading_settings": {
    "active_strategies": [
      { "symbol": "BTC/USDT:USDT", "timeframe": "1d", "active": true }
    ]
  },
  "strategy": {
    "breakout_threshold_pct": 1.5,      // Range-Grenze für Breakout-Signal
    "breakout_prob_threshold": 0.60,    // Min. Wahrscheinlichkeit für Breakout
    "range_threshold_pct": 1.0,         // Range-Grenze für Range-Signal
    "range_prob_threshold": 0.65,       // Min. Wahrscheinlichkeit für Range
    "risk_per_trade_pct": 2.0,          // Risiko pro Trade in % des Kapitals
    "leverage": 10,
    "n_simulations": 10000,             // Monte-Carlo Pfade (Backtest: 2000)
    "n_steps_intraday": 288             // Intraday-Schritte (5-Min-Raster)
  },
  "physics": {
    "garch_window": 100,
    "fractal_window": 100,
    "chaos_window": 50,
    "info_window": 30,
    "liquidity_bins": 100,
    "energy_windows": [5, 14, 30]
  }
}
```

---

## Signal-Logik

### Breakout-Signal
- `P(Range > breakout_threshold_pct) > 0.60`
- UND Phase-Space-Regime: `uptrend_attractor` oder `downtrend_attractor`
- Entry: Buy/Sell-Stop 0.1% über/unter aktuellem Preis
- TP: nächste Liquiditätszone in Richtung der Bewegung
- SL: 95%-Quantil der simulierten Low/High-Verteilung

### Range-Signal (Mean-Reversion)
- `P(Range < range_threshold_pct) > 0.65`
- UND Phase-Space-Regime: `range_zone`
- Entry: Long nahe erwartetem Low, Short nahe erwartetem High

### Kein Trade (wait)
- Lyapunov-Exponent > 0.5 (zu chaotisch)
- Fraktaldimension > 1.7 (Chaos-Regime)
- Volatilitäts-Regime: `high` + `chaotic` gleichzeitig

---

## 4D Phasenraum — Regime-Erkennung

```
(P, v=dP/dt, a=d²P/dt², ρ_L=Liquiditätsdichte)
```

| Regime | Beschreibung | Trading |
|---|---|---|
| `uptrend_attractor` | Steigende Momentum-Spirale | Breakout Long |
| `downtrend_attractor` | Fallende Momentum-Spirale | Breakout Short |
| `range_zone` | Oszillierende Preisbewegung | Mean-Reversion |
| `crash_spike_zone` | Hohe Beschleunigung + Volatilität | Wait |

---

## Mathematisches Kern-Modell

**Preisprozess (erweiterte GBM):**
```
dP_t = μ(X_t) * P_t * dt + σ(X_t) * P_t * dW_t
```

**Drift:**
```
μ = a0 + a1*Φ + a2*F_L + a3*E
```

**Effektive Volatilität:**
```
σ_eff = σ_GARCH * (1 + b1*(D-1.5) + b2*max(0,λ))
```

**Liquiditäts-Gravitationspotenzial:**
```
U(P) = -∫ G * ρ_L(P') / |P - P'| dP'
F_L  = -∇U(P)
```

---

## Cronjob (VPS)

```bash
# Alle 4 Stunden prüfen (für 1d-Timeframe)
0 */4 * * * cd /path/to/wbot && .venv/bin/python3 master_runner.py >> logs/cron.log 2>&1
```

---

## Tests

```bash
cd wbot
source .venv/bin/activate
pytest tests/ -v
```

Alle Tests laufen ohne Exchange-Verbindung (kein API-Key nötig).

---

## secret.json

```json
{
  "wbot": {
    "api_key": "...",
    "api_secret": "...",
    "passphrase": "..."
  },
  "telegram": {
    "token": "...",
    "chat_id": "..."
  }
}
```

**Nie committen** — ist in `.gitignore` ausgeschlossen.

---

## Verwandte Bots

| Bot | Ansatz |
|---|---|
| `ltbbot` | Layer-basiertes Trend-Following + Trailing-Stop |
| `mbot` | BB-Breakout + Volumen (Momentum) |
| `dnabot` | Genome-basiertes Muster-Mining (SQLite) |
| `dbot` | LSTM-Klassifikation (Long/Neutral/Short) |
| `wbot` | **QGRS: Physik-basierte Range-Prognose (Monte-Carlo)** |
