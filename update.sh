#!/bin/bash
# Sicheres Update: holt den Repo-Stand, behaelt secret.json und settings.json.
# Papierkonto, Vorhersage-Protokoll und Datencache (artifacts/live, data/world) sind nicht im Repo
# und bleiben bei git reset unangetastet.
set -e
cd "$(dirname "$0")"

echo "1. Sichere secret.json und settings.json ..."
TMP=$(mktemp -d)
cp secret.json "$TMP/secret.json"
[ -f settings.json ] && cp settings.json "$TMP/settings.json"

echo "2. Hole den neuesten Stand von GitHub ..."
git fetch origin
git reset --hard origin/main
cp "$TMP/secret.json" secret.json

echo "3. Python-Pakete in .venv aktualisieren ..."
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt

echo "4. settings.json: lokale Werte uebernehmen, neue Schluessel aus dem Repo ergaenzen ..."
.venv/bin/python3 - "$TMP/settings.json" <<'PY'
import json, os, sys
old_path = sys.argv[1]
if os.path.exists(old_path):
    old = json.load(open(old_path, encoding='utf-8')).get('live_trading_settings', {})
    new = json.load(open('settings.json', encoding='utf-8'))
    if 'mode' in old:   # nur v2-Settings uebernehmen (v1 QGRS hatte active_strategies)
        new['live_trading_settings'].update(old)
        print('   lokale settings.json beibehalten')
    else:
        print('   alte v1-settings.json verworfen, v2-Standard aus dem Repo (mode=paper)')
    json.dump(new, open('settings.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
PY
rm -rf "$TMP"

echo "5. Python-Cache leeren, Rechte setzen ..."
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
chmod +x *.sh

echo "✅ Update fertig."
