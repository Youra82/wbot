#!/bin/bash
# wbot v2 "Weltwetter" -- Erstinstallation
set -e
echo "======================================================="
echo "    wbot v2 Weltwetter -- Installation"
echo "======================================================="

echo "1/4 System-Pakete ..."
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv git

echo "2/4 Python-Umgebung (.venv) ..."
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "3/4 Rechte ..."
chmod +x *.sh
mkdir -p logs

echo "4/4 Fertig."
echo ""
echo "Naechste Schritte:"
echo "  1. secret.json anlegen (siehe secret.json.example; Telegram genuegt fuer den Papiermodus)"
echo "  2. ./run_pipeline.sh   (Daten laden + praeregistrierten Test rechnen, ca. 5 Min)"
echo "  3. Cronjob (stuendlich, laeuft nur einmal pro UTC-Tag erfolgreich):"
echo "     5 * * * * cd $(pwd) && flock -n /tmp/wbot.lock .venv/bin/python3 master_runner.py >> logs/cron.log 2>&1"
