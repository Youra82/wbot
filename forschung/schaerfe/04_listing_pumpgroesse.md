# Schärfe-Schritt 4: Upbit-Listing-Short nach Pump-Größe (Präregistrierung 2026-10-26, vor Sicht der Ergebnisse)

## Ausgangslage
1mbot 10.10.: Upbit-KRW-Listing → Binance-Perp Short ab T+30 min, 72 h: OOS +15 % netto/Ereignis (t 4,45), inkl.
Funding +9,3 %. Mit Stop/Hebel realistisch t ≈ 1,4 — Squeeze-Risiko (Zwischenverlust) frisst Signifikanz.

## Frage
Wird die Vorhersage schärfer, wenn nur Listings mit **großem Anfangs-Pump** geshortet werden?
Mechanismus: Je weiter der Kurs in 30 min übertrieben wurde, desto mehr Rückfall (Übertreibung durch
Privatanleger-Ansturm; Nachkaufdruck ist nach 10 s erschöpft, siehe 1mbot-Diagnose).

## Merkmal (eines, zum Einstieg bekannt)
pump = Open der Minute T+30 min / Open der Minute T−5 min − 1 (Binance-Perp, 1mbot/research/listing/ev).

## Regel
Schwelle = Median von pump im IS (vor 2025-07-01, wie 1mbot). Ergebnis je Ereignis = L2-netto (Kosten wie 1mbot)
+ Funding (diag_l2.csv).

## Bestanden (OOS ab 2025-07-01)
1. Gruppe pump ≥ Schwelle: Ø Ergebnis > Gruppe pump < Schwelle,
2. Gruppe pump ≥ Schwelle: Ø > 0 mit t ≥ 2,0,
3. Gruppe pump ≥ Schwelle: Median-Zwischenverlust (MAE) nicht über 1,5 × dem der anderen Gruppe (Risiko nicht explodiert).
Nur Info: Hit-Rate, Median, Anteil MAE > 50 %.
