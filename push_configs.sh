#!/bin/bash
# wbot v2 - settings.json und Testergebnis (research.json) pushen

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

RESEARCH_FILE="artifacts/results/research.json"
SETTINGS_FILE="settings.json"

echo ""
echo -e "${YELLOW}========== wbot SETTINGS PUSHEN ==========${NC}"
echo ""

# Prueffe ob settings.json existiert
if [ ! -f "$SETTINGS_FILE" ]; then
    echo -e "${RED}Fehler: settings.json nicht gefunden.${NC}"
    exit 1
fi

git add "$SETTINGS_FILE"
[ -f "$RESEARCH_FILE" ] && git add "$RESEARCH_FILE"

# Prueffe ob Aenderungen vorhanden
STAGED=$(git diff --cached --name-only)

if [ -z "$STAGED" ]; then
    echo -e "${YELLOW}Keine Aenderungen — alles ist bereits aktuell im Repo.${NC}"
    exit 0
fi

echo ""
echo "Geaenderte Dateien:"
echo "$STAGED" | sed 's/^/  /'
echo ""

# Commit
TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
git commit -m "Update: wbot settings/Testergebnis aktualisiert ($TIMESTAMP)"

# Push
echo ""
echo -e "${YELLOW}Pushe auf origin/main...${NC}"
git push origin HEAD:main

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}Configs erfolgreich gepusht!${NC}"
else
    echo ""
    echo -e "${YELLOW}Remote hat neuere Commits — fuehre Rebase durch...${NC}"
    git pull origin main --rebase
    if [ $? -ne 0 ]; then
        echo -e "${RED}Rebase fehlgeschlagen. Bitte manuell loesen.${NC}"
        exit 1
    fi
    git push origin HEAD:main
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}Configs erfolgreich gepusht!${NC}"
    else
        echo -e "${RED}Push nach Rebase fehlgeschlagen.${NC}"
        exit 1
    fi
fi
