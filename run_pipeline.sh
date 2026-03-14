#!/bin/bash
# wbot QGRS Pipeline: Backtest -> Show Results
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON=".venv/bin/python3"
VENV_PATH=".venv/bin/activate"

if [ ! -f "$VENV_PATH" ]; then
    echo -e "${RED}Fehler: Virtuelle Umgebung nicht gefunden. Bitte install.sh ausfuehren.${NC}"
    exit 1
fi

source "$VENV_PATH"

SYMBOL="${1:-BTC/USDT:USDT}"
TIMEFRAME="${2:-1d}"
START_CAPITAL="${3:-1000}"

echo -e "${BLUE}======================================================="
echo "      wbot QGRS Pipeline"
echo -e "=======================================================${NC}"
echo -e "${YELLOW}Symbol:     $SYMBOL${NC}"
echo -e "${YELLOW}Timeframe:  $TIMEFRAME${NC}"
echo -e "${YELLOW}Kapital:    $START_CAPITAL USDT${NC}"
echo ""

echo -e "${YELLOW}--- Starte Backtest ---${NC}"
PYTHONPATH="$SCRIPT_DIR/src" "$PYTHON" -m wbot.analysis.backtester \
    --symbol "$SYMBOL" \
    --timeframe "$TIMEFRAME" \
    --start-capital "$START_CAPITAL"

if [ $? -ne 0 ]; then
    echo -e "${RED}Fehler beim Backtest. Abbruch.${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}--- Zeige Ergebnisse ---${NC}"
PYTHONPATH="$SCRIPT_DIR/src" "$PYTHON" -m wbot.analysis.show_results \
    --symbol "$SYMBOL" \
    --timeframe "$TIMEFRAME"

deactivate

echo ""
echo -e "${BLUE}======================================================="
echo -e "      Pipeline abgeschlossen!"
echo -e "=======================================================${NC}"
