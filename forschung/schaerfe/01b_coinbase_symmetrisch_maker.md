# Schärfe-Schritt 1b: Coinbase stündlich, symmetrisches Ziel + Maker-Einstieg (Präregistrierung 2026-10-26, vor Sicht)

## Anlass
Schritt 1: Station kalibriert und mit echter Auflösung (Top-1 % trifft ~1,6× so oft), aber Ziel falsch gebaut
(Klima 13 %, Break-even 44 %) und Taker-Kosten fressen den Vorsprung.

## Unverändert gegenüber Schritt 1
Daten, Merkmale (z1, z4, d4 + Quadrate, nur bis t−1), Modell (LogReg C = 0,1, Long/Short getrennt, BTC+ETH gepoolt,
monatlicher Walk-Forward), Zeiträume (Validierung 2024-01 … 2025-03, Endtest 2025-04 … heute), σ = 168-h-Std. × √12.

## Neu
- **Ziel symmetrisch:** TP = SL = 1 σ, höchstens 12 h. Label = TP vor SL (vom Open der Stunde t aus).
- Break-even je Stunde: p_be = (1 + c) / 2, c = Kosten in R. Handel wenn P ≥ p_be + 3 Pkt.
- **Variante M (primär) — Maker-Einstieg:** Limit zu Beginn von Stunde t bei Open − 0,1 σ (Long) bzw. + 0,1 σ (Short),
  gültig nur Stunde t. **Ausgeführt nur, wenn das Tief (Long) bzw. Hoch (Short) der Stunde das Limit echt
  unterschreitet (strikt < bzw. >).** TP/SL relativ zum Limitpreis. In der Ausführungsstunde zählt nur ein SL, kein TP
  (konservativ). Kosten: Einstieg 0,02 %, TP-Ausstieg als Limit 0,02 %, SL/Zeit-Ausstieg Taker 0,06 % + 0,05 % Schlupf.
  Nicht ausgeführt = kein Trade. Funding wie gehabt.
- Variante T (nur Info): Market-Einstieg zum Open, 0,22 % Kosten.

## Bestanden (Variante M, in Validierung UND Endtest)
Schärfe ≥ 2 % der Stunden (P ≥ p_be + 3 Pkt) UND tatsächliche Trefferquote der ausgeführten scharfen Trades ≥ p_be + 2 Pkt
UND ≥ 50 ausgeführte Trades UND Ø R > 0 UND BSS > 0.
Hinweis: Coinbase-Prämie war schon am 23.10. und in Schritt 1 angesehen → Endtest nicht unberührt; Vorwärtsmessung entscheidet.
