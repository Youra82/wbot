# Schärfe-Schritt 1: Coinbase-Prämie stündlich (Präregistrierung 2026-10-26, vor Sicht der Ergebnisse)

## Warum
Tagesversion: Vorhersagen kleben am Klima (99 % unter 45 %). Hypothese: Die Information der Coinbase-Prämie
(US-Kaufdruck) lebt Stunden, nicht 10 Tage. Auf passendem Horizont soll sie **scharfe** Vorhersagen liefern.

## Daten
- Coinbase BTC-USD, ETH-USD, USDT-USD (1h), Binance-Spot BTCUSDT, ETHUSDT (1h), Binance-Futures BTCUSDT/ETHUSDT (1h) + Funding.
- Prämie_t = log(Coinbase-Schluss / (Binance-Spot-Schluss × Coinbase USDT-USD)) → USDT-Basis herausgerechnet.
- Zeitraum 2022-01 … heute.

## Merkmale (nur Stunden ≤ t−1, Entscheidung zu Beginn von Stunde t)
z1 = Prämie der letzten Stunde, z-normiert gegen 168 h; z4 = 4-h-Mittel, z-normiert; d4 = Änderung über 4 h, z-normiert.

## Ziel (je Coin, je Stunde)
Einstieg Open der Stunde t. σ = Std. der stündlichen Log-Renditen der letzten 168 h × √12 × Preis.
TP = 1,5 σ, SL = 1 σ, höchstens 12 h, dann Schluss. Gleiche Stunde beides → SL. Kosten 0,22 % + Funding.
Gewinn-Label = TP vor SL.

## Modell
Logistische Regression (C = 0,1) auf (z1, z4, d4) **und ihre Quadrate** (damit extreme Werte extreme P liefern),
getrennt Long/Short, BTC+ETH gepoolt, monatlicher Walk-Forward auf aufgelösten Daten (Training ab 2022).

## Messgrößen
- **Schärfe:** Anteil der Stunden mit P ≥ Break-even + 5 Punkte (Break-even je Stunde aus Kosten: (1 + c)/2,5).
- **Kalibrierung in diesem scharfen Bereich:** tatsächliche Trefferquote vs. vorhergesagte.
- BSS gesamt; Handel: Seite mit höherem P wenn P ≥ Break-even + 5 Pkt, eine Position je Coin.

## Zeiträume
Validierung 2024-01 … 2025-03, Endtest 2025-04 … heute.
**Achtung:** Die BTC-Stundenprämie wurde am 23.10. schon über 2019–2026 angesehen (Filter für Long-Halten). Der
Endtest ist daher nicht völlig unberührt; endgültiger Richter ist die Vorwärtsmessung im Papiermodus.

## Bestanden (beide Zeiträume)
Schärfe ≥ 2 % der Stunden UND im scharfen Bereich tatsächliche Trefferquote ≥ Break-even + 2 Pkt UND
Handel Ø R > 0 bei ≥ 50 Trades UND BSS > 0.
