#!/bin/bash
set -e

echo "--- wbot QGRS - Sicheres Update ---"

# 1. secret.json sichern
echo "1. Erstelle Backup von 'secret.json'..."
cp secret.json secret.json.bak

# 2. Neuesten Stand von GitHub holen
echo "2. Hole den neuesten Stand von GitHub..."
git fetch origin

# 3. Lokal auf GitHub-Stand zuruecksetzen
echo "3. Setze alle Dateien auf den neuesten Stand zurueck..."
git reset --hard origin/main

# 4. secret.json wiederherstellen
echo "4. Stelle 'secret.json' aus dem Backup wieder her..."
cp secret.json.bak secret.json
rm secret.json.bak

# 5. Python-Cache loeschen
echo "5. Loesche alten Python-Cache..."
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -delete

# 6. Ausfuehrungsrechte setzen
echo "6. Setze Ausfuehrungsrechte fuer alle .sh-Skripte..."
chmod +x *.sh

echo "Update erfolgreich abgeschlossen."
