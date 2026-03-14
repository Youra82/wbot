#!/bin/bash
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

VENV_PATH=".venv"
VENV_ACTIVATE="$VENV_PATH/bin/activate"
VENV_PYTHON="$VENV_PATH/bin/python3"

if ! test -x "$VENV_PYTHON"; then
    echo -e "${RED}Fehler: Virtuelle Umgebung nicht gefunden. Bitte install.sh ausfuehren.${NC}"
    exit 1
fi

if ! source "$VENV_ACTIVATE"; then
    echo -e "${RED}Fehler beim Aktivieren der venv!${NC}"
    exit 1
fi

SYMBOL="${1:-BTC/USDT:USDT}"
TIMEFRAME="${2:-1d}"

echo ""
echo -e "${BLUE}======================================================="
echo "      wbot QGRS - Ergebnisse & Forecast"
echo -e "=======================================================${NC}"
echo -e "${YELLOW}Symbol:    $SYMBOL${NC}"
echo -e "${YELLOW}Timeframe: $TIMEFRAME${NC}"
echo ""

echo -e "${YELLOW}Waehle Anzeigemodus:${NC}"
echo "  1) Backtest-Ergebnisse aus letztem Run"
echo "  2) Live Forecast (aktuelle Marktdaten)"
echo "  3) Phase-Space Analyse"
read -p "Auswahl (1-3) [Standard: 1]: " MODE
MODE=${MODE:-1}

if [[ ! "$MODE" =~ ^[1-3]$ ]]; then
    echo -e "${RED}Ungueltige Eingabe — verwende Standard (1).${NC}"
    MODE=1
fi

echo ""
export PYTHONPATH="$SCRIPT_DIR/src"
"$VENV_PYTHON" -m wbot.analysis.show_results \
    --symbol "$SYMBOL" \
    --timeframe "$TIMEFRAME" \
    --mode "$MODE"

if command -v deactivate &> /dev/null; then
    deactivate
fi
