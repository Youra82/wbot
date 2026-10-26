# Schärfe-Schritt 2: Korea-Prämie stündlich (Präregistrierung 2026-10-26, vor Sicht der Ergebnisse)

## Mechanismus
Upbit (KRW) ist vom Rest der Welt teilweise abgeschottet (Kapitalverkehrskontrollen). Koreanische Privatanleger
treiben dort eigene Pumps/Dips. Eine plötzlich hohe Prämie = lokaler Kaufdruck, der über Arbitrage-Desks mit
Verzögerung auf Binance überschwappt (Fortsetzung) — oder sich als Übertreibung zurückbildet (Umkehr). Das Modell
lernt die Richtung; Quadrate erlauben beides an den Extremen.

## Daten
Upbit KRW-BTC, KRW-ETH, KRW-XRP, KRW-SOL, KRW-DOGE (60-min-Kerzen), Binance-Spot gleiche Coins (1h),
Binance-Futures (1h) + Funding. 2022-01 … heute.
Prämie_t = log(Upbit-Schluss / Binance-Spot-Schluss); Abweichung = Prämie − 168-h-Mittel (Wechselkurs fällt heraus).

## Merkmale (nur bis t−1)
Je Coin: k1 = Abweichung z (168 h), k4 = 4-h-Mittel z, kd4 = Änderung 4 h z; Markt: kbtc1 = BTC-Abweichung z.
Dazu alle Quadrate.

## Ziel, Modell, Zeiträume
Wie Schritt 1b: TP = SL = 1 σ (σ = 168-h-Std. × √12), höchstens 12 h; Label = TP vor SL; LogReg C = 0,1, Long/Short
getrennt, 5 Coins gepoolt, monatlicher Walk-Forward. Validierung 2024-01 … 2025-03, Endtest 2025-04 … heute.
Maker-Variante wie 1b (Limit Open ∓ 0,1 σ, nur bei echtem Durchschreiten; Kosten 0,02 % / 0,02 % TP bzw. 0,11 % SL/Zeit).

## Kriterium (schwellenfrei, aus Schritt 1b gelernt)
Signal = P liegt über dem 90 %-Quantil der Vorhersagen **im Trainingszeitraum des Monats** (kausal, keine Kenntnis
der Testverteilung). Bestanden, wenn in Validierung UND Endtest:
1. Ø R (Maker, nur ausgeführte) aller Signal-Stunden > 0,
2. Handel (eine Position je Coin, Seite mit höherem P relativ zu ihrem Quantil) Ø R > 0 bei ≥ 50 Trades,
3. BSS > 0.
Diese Station wurde noch nie stündlich angesehen (Tagesversion am 26.10. in der Validierung BSS < 0).
