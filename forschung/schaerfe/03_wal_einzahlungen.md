# Schärfe-Schritt 3: Wal-Einzahlungen schärfer machen (Präregistrierung 2026-10-26, vor Sicht der Ergebnisse)

## Ausgangslage
kbot 09.10.: Einzahlung ≥ 500 k $ eines ERC-20-Alts an eine bekannte Binance-Einzahladresse → Short 60 min:
+0,2 % brutto vs. Basis (IS t 2,9, OOS t 2,3), netto nach 0,12 % Taker nur +0,03 % (OOS). Effekt echt, aber zu flach.

## Frage
Lassen sich **vor dem Einstieg bekannte** Eigenschaften der Einzahlung nutzen, um die Vorhersage zu schärfen:
ein Teil der Ereignisse mit deutlich besserem Ergebnis als der Durchschnitt?

## Daten und Regel (unverändert aus kbot, gleiche Funktionen wie dessen Live-Wächter)
kbot/research/cex/data (22 Alts, Minuten-Schlusskurse, Okt. 2025 … Okt. 2026), `DepositRule`, `entry_minute`.
Ereignisse: Einzahlung ≥ **250 k $** (mehr Stichprobe), Short vom Einstiegsminuten-Schluss 60 min, Kosten 0,12 %.
IS = bis 2026-06-08, OOS = ab 2026-06-09 (wie kbot).

## Merkmale (alle zum Einstieg bekannt)
1. log(Betrag in $)
2. vor60 = Kursänderung des Tokens in den 60 min bis zum Einstieg (Mechanismus: Wale zahlen nach Anstieg ein)
3. wal = Absender ist keine bekannte Börsen-Wallet
4. cluster = Anzahl weiterer Einzahlungen ≥ 250 k $ desselben Tokens in den 60 min davor
5. btc60 = BTC-Kursänderung in den 60 min bis zum Einstieg

## Modell
Ridge-Regression (α = 1, standardisiert) der Short-Rendite (60 min) auf die 5 Merkmale, **nur auf IS** angepasst,
dann OOS vorhergesagt. Kein Nachtrainieren, keine Merkmalsauswahl.

## Bestanden (OOS)
1. oberste 30 % der Vorhersagen: Ø Short netto (nach 0,12 %) > 0 mit t ≥ 1,5,
2. oberste 30 % besser als alle OOS-Ereignisse,
3. Spearman-Korrelation Vorhersage ↔ Ergebnis > 0.

Hinweis: Absender-Typ (wal) wurde in kbot schon auf IS+OOS beschreibend angesehen → leicht vorbelastet.
