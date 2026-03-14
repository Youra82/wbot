# src/wbot/utils/trade_manager.py
# Live-Order-Verwaltung: Entry, SL, TP, Positionen
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

MIN_NOTIONAL_USDT = 5.0  # Mindest-Positionsgroesse in USDT


def get_open_positions(exchange, symbol: str) -> List[Dict]:
    """
    Fragt offene Positionen fuer ein Symbol von der Exchange ab.

    Args:
        exchange: ccxt Exchange-Objekt
        symbol: Handelspaar (z.B. "BTC/USDT:USDT")

    Returns:
        Liste von Position-Dicts (leere Liste wenn keine offene Position)
    """
    try:
        positions = exchange.fetch_positions([symbol])
        open_positions = []

        for pos in positions:
            contracts = float(pos.get('contracts', 0) or 0)
            if contracts > 0:
                open_positions.append({
                    'symbol': pos.get('symbol'),
                    'side': pos.get('side'),
                    'contracts': contracts,
                    'entry_price': pos.get('entryPrice'),
                    'unrealized_pnl': pos.get('unrealizedPnl'),
                    'leverage': pos.get('leverage'),
                    'notional': pos.get('notional'),
                })

        logger.debug(f"Offene Positionen fuer {symbol}: {len(open_positions)}")
        return open_positions

    except Exception as e:
        logger.error(f"Positions-Abfrage fehlgeschlagen fuer {symbol}: {e}")
        return []


def calculate_position_size(
    capital: float,
    risk_pct: float,
    entry: float,
    stop_loss: float,
    leverage: float = 10.0
) -> float:
    """
    Berechnet die Positionsgroesse in USDT.

    Logik: Risikobetrag / Stop-Distance als Anteil, skaliert mit Hebel.

    Args:
        capital: Verfuegbares Kapital in USDT
        risk_pct: Risiko in % des Kapitals pro Trade
        entry: Einstiegspreis
        stop_loss: Stop-Loss-Preis
        leverage: Hebel-Faktor

    Returns:
        size_usdt - Positionsgroesse in USDT (Notional)
    """
    try:
        if entry <= 0 or stop_loss <= 0 or capital <= 0:
            return MIN_NOTIONAL_USDT

        risk_amount = capital * (risk_pct / 100.0)
        stop_distance_pct = abs(entry - stop_loss) / entry

        if stop_distance_pct < 0.0001:
            logger.warning("Stop-Distance zu klein fuer Positionsgroessen-Berechnung.")
            return MIN_NOTIONAL_USDT

        # Margin = Risiko / Stop-Distance
        margin_needed = risk_amount / stop_distance_pct
        # Notional = Margin * Leverage
        size_usdt = margin_needed * leverage

        # Minimum sicherstellen
        size_usdt = max(size_usdt, MIN_NOTIONAL_USDT)

        logger.debug(
            f"Positionsgroesse: {size_usdt:.2f} USDT | "
            f"risk={risk_amount:.2f} USDT | stop_dist={stop_distance_pct:.4f} | "
            f"leverage={leverage}x"
        )
        return float(size_usdt)

    except Exception as e:
        logger.warning(f"Positionsgroessen-Berechnung fehlgeschlagen: {e}")
        return MIN_NOTIONAL_USDT


def place_entry_order(
    exchange,
    symbol: str,
    side: str,
    size_usdt: float,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    leverage: int = 10
) -> Optional[Dict]:
    """
    Platziert eine Entry-Order mit Stop-Loss und Take-Profit.

    Nutzt Bitget-spezifische Parameter fuer SL/TP.

    Args:
        exchange: ccxt Bitget Exchange-Objekt
        symbol: Handelspaar
        side: "buy" oder "sell"
        size_usdt: Positionsgroesse in USDT (Notional)
        entry_price: Gewuenschter Einstiegspreis (Limit)
        sl_price: Stop-Loss-Preis
        tp_price: Take-Profit-Preis
        leverage: Hebel-Faktor

    Returns:
        Order-Dict oder None bei Fehler
    """
    try:
        # Mindest-Notional pruefen
        if size_usdt < MIN_NOTIONAL_USDT:
            logger.warning(f"Notional {size_usdt:.2f} USDT < Minimum {MIN_NOTIONAL_USDT} USDT. Kein Trade.")
            return None

        # Leverage setzen
        try:
            exchange.set_leverage(leverage, symbol)
            logger.info(f"Leverage gesetzt: {leverage}x fuer {symbol}")
        except Exception as e:
            logger.warning(f"Leverage-Setzung fehlgeschlagen (nicht kritisch): {e}")

        # Asset-Menge berechnen: contracts = notional / entry_price
        amount = size_usdt / entry_price
        logger.info(
            f"Order-Parameter: {side} {amount:.6f} contracts @ {entry_price:.4f} | "
            f"SL={sl_price:.4f} | TP={tp_price:.4f} | leverage={leverage}x"
        )

        # Order platzieren (Limit-Order mit SL/TP als Parameter)
        order = exchange.create_order(
            symbol=symbol,
            type='limit',
            side=side,
            amount=amount,
            price=entry_price,
            params={
                'stopLoss': {'triggerPrice': sl_price, 'type': 'market'},
                'takeProfit': {'triggerPrice': tp_price, 'type': 'market'},
                'reduceOnly': False,
            }
        )

        logger.info(f"Order platziert: ID={order.get('id')} | Status={order.get('status')}")
        return order

    except Exception as e:
        logger.error(f"Order-Platzierung fehlgeschlagen: {e}", exc_info=True)
        return None


def close_position(exchange, symbol: str, side: str) -> bool:
    """
    Schliesst eine offene Position per Market-Order.

    Args:
        exchange: ccxt Exchange-Objekt
        symbol: Handelspaar
        side: "long" oder "short" (bestehende Position-Seite)

    Returns:
        True bei Erfolg, False bei Fehler
    """
    try:
        # Gegenrichtung fuer Close
        close_side = 'sell' if side == 'long' else 'buy'

        # Aktuell offene Position abfragen
        positions = get_open_positions(exchange, symbol)
        if not positions:
            logger.info(f"Keine offene Position zum Schliessen fuer {symbol}.")
            return True

        for pos in positions:
            if pos['side'] == side or pos['side'] == ('long' if side == 'buy' else 'short'):
                contracts = pos['contracts']
                if contracts > 0:
                    order = exchange.create_order(
                        symbol=symbol,
                        type='market',
                        side=close_side,
                        amount=contracts,
                        params={'reduceOnly': True}
                    )
                    logger.info(f"Position geschlossen: {symbol} {side} | {contracts} contracts")
                    return True

        logger.warning(f"Keine passende Position zum Schliessen gefunden fuer {symbol} ({side}).")
        return False

    except Exception as e:
        logger.error(f"Position-Schliessung fehlgeschlagen fuer {symbol}: {e}", exc_info=True)
        return False
