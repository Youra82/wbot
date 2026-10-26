#!/bin/bash
# wbot v2: Testergebnis, Papierkonto und letzte Vorhersage anzeigen
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] || { echo "Fehler: .venv fehlt -- erst ./install.sh"; exit 1; }
source .venv/bin/activate
PYTHONPATH=src python3 src/wbot/analysis/show_results.py
deactivate
