#!/bin/bash
set -e
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

echo -e "${YELLOW}=== wbot QGRS Installation ===${NC}"

# 1. venv erstellen
echo -e "${YELLOW}1. Erstelle virtuelle Umgebung (.venv)...${NC}"
python3 -m venv .venv --upgrade-deps
echo -e "${GREEN}   Virtuelle Umgebung erstellt.${NC}"

# 2. pip upgrade
echo -e "${YELLOW}2. Upgrade pip...${NC}"
.venv/bin/pip install --upgrade pip setuptools wheel --quiet

# 3. Requirements installieren
echo -e "${YELLOW}3. Installiere Abhaengigkeiten (requirements.txt)...${NC}"
.venv/bin/pip install -r requirements.txt
echo -e "${GREEN}   Abhaengigkeiten installiert.${NC}"

# 4. Verzeichnisse anlegen
echo -e "${YELLOW}4. Erstelle fehlende Verzeichnisse...${NC}"
mkdir -p artifacts/results artifacts/logs logs tests
echo -e "${GREEN}   Verzeichnisse vorhanden.${NC}"

# 5. secret.json aus Template
if [ ! -f "secret.json" ]; then
    echo -e "${YELLOW}5. Erstelle secret.json aus Template...${NC}"
    cp secret.json.example secret.json
    echo -e "${GREEN}   secret.json erstellt - bitte API-Keys eintragen!${NC}"
else
    echo -e "${GREEN}5. secret.json bereits vorhanden.${NC}"
fi

# 6. Ausfuehrungsrechte
echo -e "${YELLOW}6. Setze Ausfuehrungsrechte...${NC}"
chmod +x *.sh

# 7. Import-Test
echo -e "${YELLOW}7. Pruefe Installation...${NC}"
if PYTHONPATH="$SCRIPT_DIR/src" .venv/bin/python3 -c "
from wbot.physics.garch_volatility import estimate_garch
from wbot.physics.fractal_dimension import hurst_exponent
from wbot.model.monte_carlo import run_monte_carlo
print('   wbot Module OK')
" 2>/dev/null; then
    echo -e "${GREEN}   Alle Module erfolgreich importiert.${NC}"
else
    echo -e "${RED}   Import-Test fehlgeschlagen. Bitte Fehler pruefen.${NC}"
fi

echo ""
echo -e "${GREEN}=== Installation abgeschlossen ===${NC}"
echo ""
echo "Naechste Schritte:"
echo "  1. secret.json bearbeiten:  nano secret.json"
echo "  2. Backtest starten:        ./run_pipeline.sh"
echo "  3. Live-Trading:            .venv/bin/python3 master_runner.py"
