# wbot v2 „Weltwetter“ — Präregistrierung (2026-10-26, vor Sicht der Ergebnisse)

## Idee
Wie ein Wetterdienst: viele unabhängige Messstationen aus der Welt statt Indikatoren auf dem eigenen Kurs.
Jede Station ist ein eigenes kleines Vorhersagemodell. Nur Stationen, die einzeln Können zeigen,
gehen ins Ensemble. Das Ensemble kombiniert die Abweichungen der Stationen vom Klima (Grundhäufigkeit).

## Universum, Zeitpunkt, Trade
- Coins: BTC, ETH, SOL, XRP, DOGE, BNB (USDT-Perps). Kurse/Funding für Forschung + Live-Merkmale: Binance-Futures
  (öffentlich, lange Historie), Ausführung Bitget.
- Entscheidung einmal täglich zu Beginn von Tag t (00:05 UTC), nur mit Daten, die bis dahin veröffentlicht sind.
- Einstieg Open[t]. ATR(14) aus Kerzen ≤ t−1. TP = 1,5 × ATR, SL = 1 × ATR, max. 10 Tage, dann Schluss zum Close.
  SL und TP in derselben Kerze → SL. Kosten 0,22 % pro Runde + tatsächliches Binance-Funding während der Haltezeit.
- Eine Position je Coin gleichzeitig.

## Stationen (Merkmale, Verzögerung)
| Station | Quelle | Mechanismus |
|---|---|---|
| preis | eigene Kurse (Renditen 1/3/10/30 T vola-normiert, Lage im 20-T-Band, Vola-Verhältnis, Abstand MA50) | Referenz/„aktueller Zustand“ |
| makro | Yahoo: Nasdaq-Fut., Dollar-Index, US-10J-Zins, Gold, VIX (1-T/5-T-Änderung) | Risikoappetit der Welt, Krypto hängt am Nasdaq |
| stablecoins | DefiLlama Gesamtmenge (7-T/30-T-Änderung z), +1 Tag Verzögerung | frisches Kapital |
| coinbase | Coinbase BTC-USD/ETH-USD vs. Binance (Prämie, 1-T und 7-T-Mittel) | US-Nachfrage |
| korea | Upbit KRW-BTC/KRW-ETH vs. Binance × USD/KRW | asiatische Privatanleger |
| aufmerksamkeit | Wikipedia-Aufrufe (Coin-Artikel + „Bitcoin“), z vs. 30 T, +1 Tag Verzögerung | Privatanleger kommen verzögert |
| derivate | Binance-Funding je Coin (Tagesmittel, 7-T-Mittel, z) | Hebel-Überhang |

Nicht aufgenommen: Token-Unlocks (keine freie Historie, für diese 6 Coins kaum relevant), Google Trends
(nicht reproduzierbar, Wikipedia ersetzt es).

## Modell
Je Station zwei logistische Regressionen (C = 0,1, standardisierte Merkmale, über alle Coins gepoolt):
P(Long-Ziel vor Stop), P(Short-Ziel vor Stop). Monatliches Nachtrainieren auf allen bis Monatsbeginn
**aufgelösten** Beobachtungen (Walk-Forward), Live identisch.
Handel einer Station/des Ensembles: Seite mit höherer Wahrscheinlichkeit, wenn P ≥ 0,45 (Break-even ≈ 0,42).
Ensemble: logit(P) = logit(Klima) + Σ_Stationen [logit(P_i) − logit(Klima)].

## Zeiträume
- Lernen ab Datenbeginn (2020), Vorhersagen ab 2023-01 (≥ 1 Jahr aufgelöste Daten).
- Validierung (Stationsauswahl): 2024-01-01 … 2025-03-31.
- Endgültiger Test (Ensemble, vorher nicht angesehen): 2025-04-01 … heute.

## Kriterien
Station aufgenommen, wenn in der Validierung: BSS > 0 für Long oder Short UND eigener Handel Ø R > 0 bei ≥ 30 Trades.
Ensemble bestanden, wenn im Test: mittlerer BSS (Long/Short) > 0, PF ≥ 1,2, Ø ≥ +0,1 R/Trade, ≥ 4/6 Coins positiv,
≥ 100 Trades/Jahr.

## Konsequenz
- Bestanden → Bot läuft zuerst im Papiermodus (Vorwärtstest 8 Wochen), Echtgeld erst danach auf User-Entscheid.
- Nicht bestanden → Bot ist vollständig gebaut, läuft nur im Papiermodus (sammelt Vorwärtsdaten), keine Echtgeld-Freigabe.
- Keine nachträgliche Änderung von Schwellen, Stationen, TP/SL oder Zeiträumen. Abweichungen werden hier als Nachtrag vermerkt.

## Nachtrag 1 (2026-10-26, vor dem Testlauf)
Yahoo blockt automatisierte Abrufe → Makro-Station ersetzt: Nasdaq, VIX, US-10J aus FRED (St. Louis Fed) mit
**+2 Tagen** Verzögerung (FRED veröffentlicht 1–2 Tage später), Gold = Binance PAXG, Dollar = Binance EUR/USDT.
Korea-Prämie als Abweichung vom eigenen 30-Tage-Mittel (Wechselkursdrift fällt heraus; Upbit USDT-KRW erst ab 2024).
Merkmalsliste der Stationen sonst wie oben; endgültige Definition in `src/wbot/world/features.py`.

## Ergebnis (2026-10-26) — NICHT BESTANDEN
Klima: Long-Ziel 35 %, Short-Ziel 31 %. Die Stationsvorhersagen weichen kaum davon ab (99 %-Quantil 0,42–0,46).

| Station | Validierung BSS Long / Short | Trades | Ø R | aufgenommen |
|---|---|---|---|---|
| preis | +0,0034 / +0,0162 | 53 | +0,068 | ja |
| makro | −0,0024 / −0,0147 | 12 | −0,897 | nein |
| stablecoins | −0,0054 / −0,0059 | 0 | – | nein |
| coinbase | +0,0024 / +0,0003 | 12 | +0,839 | nein (< 30 Trades) |
| korea | −0,0062 / −0,0035 | 0 | – | nein |
| aufmerksamkeit | −0,0068 / −0,0057 | 8 | −0,066 | nein |
| derivate | +0,0097 / −0,0004 | 17 | +0,095 | nein (< 30 Trades) |

Endtest 2025-04 … 2026-10 (Ensemble = preis): BSS −0,0023, 45 Trades (29/J), Ø −0,016 R, PF 0,97, 2/6 Coins → nicht bestanden.
Nur Info: Coinbase hat auch im Endtest BSS +0,0049 (positiv in beiden Zeiträumen, passt zum Befund vom 23.10.), aber nur 6 Trades.

Konsequenz laut Präregistrierung: Bot läuft nur im Papiermodus und misst alle Stationen vorwärts.
