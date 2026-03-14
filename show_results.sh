#!/bin/bash
# wbot QGRS Ergebnis-Analyse (4 Modi)

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PATH=".venv/bin/activate"
RESULTS_SCRIPT="src/wbot/analysis/show_results.py"

if [ ! -f "$VENV_PATH" ]; then
    echo -e "${RED}Fehler: Virtuelle Umgebung nicht gefunden. Bitte install.sh ausfuehren.${NC}"
    exit 1
fi
source "$VENV_PATH"

if [ ! -f "$RESULTS_SCRIPT" ]; then
    echo -e "${RED}Fehler: Analyse-Skript '$RESULTS_SCRIPT' nicht gefunden.${NC}"
    deactivate
    exit 1
fi

echo -e "\n${YELLOW}Waehle einen Analyse-Modus:${NC}"
echo "  1) Einzel-Analyse (jede Strategie wird isoliert getestet)"
echo "  2) Manuelle Portfolio-Simulation (du waehlst das Team)"
echo "  3) Automatische Portfolio-Optimierung (der Bot waehlt das beste Team)"
echo "  4) Interaktive Charts (Equity-Kurve je Strategie)"
read -p "Auswahl (1-4) [Standard: 1]: " MODE

# Validierung
if [[ ! "$MODE" =~ ^[1-4]?$ ]]; then
    echo -e "${RED}Ungueltige Eingabe. Verwende Standard (1).${NC}"
    MODE=1
fi
MODE=${MODE:-1}

# Max Drawdown fuer Modus 3
TARGET_MAX_DD=30
if [ "$MODE" == "3" ]; then
    read -p "Gewuenschter maximaler Drawdown in % [Standard: 30]: " DD_INPUT
    if [[ "$DD_INPUT" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        TARGET_MAX_DD=$DD_INPUT
    else
        echo "Ungueltige Eingabe, verwende Standard: ${TARGET_MAX_DD}%"
    fi
fi

PYTHONPATH="src" python3 "$RESULTS_SCRIPT" --mode "$MODE" --target_max_drawdown "$TARGET_MAX_DD"
RESULT_CODE=$?

# Modus 4: Abschluss-Meldung
if [ "$MODE" == "4" ]; then
    if [ $RESULT_CODE -eq 0 ]; then
        echo -e "${GREEN}Charts wurden generiert (artifacts/charts/).${NC}"
    else
        echo -e "${RED}Fehler beim Generieren der Charts.${NC}"
    fi
    deactivate
    exit $RESULT_CODE
fi

# Modus 3: Optional settings.json aktualisieren
if [ "$MODE" == "3" ]; then
    echo ""
    echo -e "${YELLOW}-------------------------------------------------${NC}"
    read -p "Sollen die optimalen Strategien in settings.json eingetragen werden? (j/n): " AUTO_UPDATE
    AUTO_UPDATE="${AUTO_UPDATE//[$'\r\n ']/}"

    if [[ "$AUTO_UPDATE" == "j" || "$AUTO_UPDATE" == "J" || "$AUTO_UPDATE" == "y" || "$AUTO_UPDATE" == "Y" ]]; then
        OPTIMIZATION_FILE="artifacts/results/optimization_results.json"

        if [ ! -f "$OPTIMIZATION_FILE" ]; then
            echo -e "${RED}Fehler: optimization_results.json nicht gefunden.${NC}"
        else
            echo -e "${BLUE}Uebertrage Ergebnisse in settings.json...${NC}"

            python3 << 'EOF'
import json
import re

with open('artifacts/results/optimization_results.json', 'r') as f:
    opt_results = json.load(f)

optimal_configs = opt_results.get('optimal_portfolio', [])

if not optimal_configs:
    print("Kein optimales Portfolio gefunden. settings.json bleibt unveraendert.")
else:
    strategies = []
    for config_name in optimal_configs:
        match = re.match(r'config_([A-Z0-9]+)USDTUSDT_(\w+)\.json', config_name)
        if match:
            coin      = match.group(1)
            timeframe = match.group(2)
            strategies.append({
                "symbol":    f"{coin}/USDT:USDT",
                "timeframe": timeframe,
                "active":    True
            })

    with open('settings.json', 'r') as f:
        settings = json.load(f)

    settings['live_trading_settings']['active_strategies'] = strategies

    with open('settings.json', 'w') as f:
        json.dump(settings, f, indent=4)

    print(f"{len(strategies)} Strategien wurden in settings.json eingetragen:")
    for strat in strategies:
        print(f"   - {strat['symbol']} ({strat['timeframe']})")
EOF

            echo -e "${GREEN}settings.json erfolgreich aktualisiert.${NC}"
        fi
    else
        echo -e "${YELLOW}Keine Aenderungen an settings.json vorgenommen.${NC}"
    fi
fi

deactivate
