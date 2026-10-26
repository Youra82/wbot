#!/bin/bash
# wbot v2: Daten aller Wetterstationen laden + praeregistrierten Test rechnen (forschung/PRAEREGISTRIERUNG.md)
set -e
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] || { echo "Fehler: .venv fehlt -- erst ./install.sh"; exit 1; }
source .venv/bin/activate
echo "1/2 Wetterstationen aktualisieren ..."
PYTHONPATH=src python3 -m wbot.world.sources
echo "2/2 Walk-Forward-Test (Validierung + Endtest) ..."
PYTHONPATH=src python3 src/wbot/analysis/research.py
deactivate
echo "Fertig. Ergebnis: ./show_results.sh"
