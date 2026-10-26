# Schärfe-Schritt 5: Funding im Querschnitt, marktneutral (Präregistrierung 2026-10-26, vor Sicht der Ergebnisse)

## Abgrenzung (nicht wiederbeleben, was beendet ist)
Ausgeschlossen und NICHT erneut getestet: Funding als Zeitreihen-Signal je Coin (Extrem-Ereignisse 23.10.,
Funding-z als lineares Merkmal 1h/4h im Dark-Matter-Test 23.10.). Diese Tests scheiterten bzw. waren Marktbeta.
Neu: **Rangfolge zwischen Coins am selben Tag**, Long und Short gleichzeitig → Marktrichtung fällt heraus.

## Mechanismus
Hohes Funding = überfüllte Long-Seite mit Hebel. Short darauf kassiert Funding (Carry) und profitiert, wenn sich die
Überfüllung relativ zu anderen Coins abbaut. Niedriges/negatives Funding umgekehrt.

## Universum
40 liquide Binance-USDT-Perps (fest, heute existierend → Survivorship-Verzerrung bewusst in Kauf, kleiner Effekt
auf Querschnitt): BTC ETH BNB SOL XRP DOGE ADA AVAX LINK DOT LTC BCH TRX ETC ATOM NEAR FIL APT ARB OP INJ SUI AAVE
UNI XLM HBAR ICP SAND MANA AXS CRV LDO RUNE GALA CHZ DYDX FET GRT EGLD THETA.
Ein Coin zählt ab 60 Tagen Kurs- und Funding-Historie.

## Regel (keine angepassten Parameter)
Täglich 00:05 UTC: Signal = Mittel der Funding-Raten der letzten 3 Tage (9 Zahlungen, alle vor 00:00 bezahlt).
Short das oberste Fünftel, Long das unterste Fünftel, gleich gewichtet, je Seite 50 % des Kapitals, Haltedauer 1 Tag
(Open → Open der Tageskerze). Ertrag inkl. tatsächlich gezahltem/erhaltenem Funding des Tages. Kosten 0,11 % je Seite auf
den Umsatz (Positionsänderungen).

## Zeiträume
Lernen 2021–2023 (nur Info), Validierung 2024-01 … 2025-03, Endtest 2025-04 … heute.

## Bestanden (Validierung UND Endtest, nach Kosten)
Ø Tagesertrag > 0 mit t ≥ 2,0 UND Sharpe (annualisiert) ≥ 1,0 UND |Beta zu BTC| ≤ 0,3.
