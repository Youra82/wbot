# src/wbot/strategy/signal_logic.py
# Entry/Exit-Logik aus Range-Forecast und MarketState
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Trading-Signal mit Entry, SL, TP und Metadaten."""
    action: str              # "long_breakout" | "short_breakout" | "long_range" | "short_range" | "wait"
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float  # % des Kapitals
    confidence: float         # 0-1
    reason: str


def _find_nearest_attractor(attractors: list, current_price: float, direction: str) -> float:
    """
    Findet den naechsten Liquiditaets-Attraktor in der angegebenen Richtung.

    Args:
        attractors: Liste von (preis, staerke) Tupeln
        current_price: Aktueller Preis
        direction: "up" oder "down"

    Returns:
        Preis des naechsten Attraktors, oder Fallback-Wert
    """
    if not attractors:
        if direction == "up":
            return current_price * 1.02
        else:
            return current_price * 0.98

    candidates = []
    for price, strength in attractors:
        if direction == "up" and price > current_price:
            candidates.append((price, strength))
        elif direction == "down" and price < current_price:
            candidates.append((price, strength))

    if not candidates:
        # Fallback wenn kein Attraktor in der Richtung
        if direction == "up":
            return current_price * 1.02
        else:
            return current_price * 0.98

    # Naechster Attraktor nach Preis-Abstand (nicht Staerke)
    nearest = min(candidates, key=lambda x: abs(x[0] - current_price))
    return nearest[0]


def _compute_position_size(
    risk_per_trade_pct: float,
    entry_price: float,
    stop_loss_price: float,
    leverage: float = 1.0,
    min_pct: float = 1.0,
    max_pct: float = 95.0
) -> float:
    """
    Berechnet die Positionsgroesse basierend auf Risiko und Stop-Distance.

    size_pct = risk_per_trade_pct / stop_distance_pct (Kelly-aehnlich)

    Args:
        risk_per_trade_pct: Maximales Risiko pro Trade in % des Kapitals
        entry_price: Einstiegspreis
        stop_loss_price: Stop-Loss-Preis
        leverage: Hebel
        min_pct: Minimum Positionsgroesse in %
        max_pct: Maximum Positionsgroesse in %

    Returns:
        position_size_pct (Anteil des Kapitals in %)
    """
    try:
        if entry_price <= 0 or stop_loss_price <= 0:
            return min_pct

        stop_distance_pct = abs(entry_price - stop_loss_price) / entry_price * 100.0

        if stop_distance_pct < 0.01:
            return min_pct

        # Kelly-aehnliche Groessenkalkulation
        size_pct = (risk_per_trade_pct / stop_distance_pct) * 100.0

        # Beruecksichtige Hebel
        size_pct = size_pct / leverage if leverage > 0 else size_pct

        # Caps anwenden
        size_pct = float(np.clip(size_pct, min_pct, max_pct))
        return size_pct

    except Exception as e:
        logger.warning(f"Positionsgroessen-Berechnung fehlgeschlagen: {e}")
        return min_pct


def generate_signal(
    forecast,
    state,
    current_price: float,
    config: dict
) -> TradingSignal:
    """
    Generiert ein Trading-Signal aus Range-Forecast und MarketState.

    Logik-Hierarchie:
    1. Warte-Bedingungen (kein Trade)
    2. Breakout-Signal (grosse Range erwartet + Trend-Regime)
    3. Range-Signal (kleine Range erwartet + Range-Regime)
    4. Warte-Signal (keine Bedingung erfuellt)

    Args:
        forecast: RangeForecast Objekt
        state: MarketState Objekt
        current_price: Aktueller Preis
        config: Strategie-Konfiguration (strategy-Sektion aus settings.json)

    Returns:
        TradingSignal
    """
    breakout_threshold_pct = config.get('breakout_threshold_pct', 1.5)
    breakout_prob_threshold = config.get('breakout_prob_threshold', 0.60)
    range_threshold_pct = config.get('range_threshold_pct', 1.0)
    range_prob_threshold = config.get('range_prob_threshold', 0.65)
    risk_per_trade_pct = config.get('risk_per_trade_pct', 2.0)
    leverage = config.get('leverage', 10)

    wait_signal = TradingSignal(
        action="wait",
        entry_price=current_price,
        stop_loss=current_price,
        take_profit=current_price,
        position_size_pct=0.0,
        confidence=0.0,
        reason="Keine Bedingung erfuellt"
    )

    try:
        # --- 1. WARTE-BEDINGUNGEN ---

        # Chaos + hoher Lyapunov = kein Trade
        if state.attractor_regime == "chaotic" and state.lyapunov > 0.5:
            logger.info("WAIT: Chaotisches Regime mit hohem Lyapunov-Exponent.")
            return TradingSignal(
                action="wait",
                entry_price=current_price,
                stop_loss=current_price,
                take_profit=current_price,
                position_size_pct=0.0,
                confidence=0.0,
                reason=f"Chaotisches Regime (lambda={state.lyapunov:.3f})"
            )

        # Hohe Volatilitaet + hohe Fraktaldimension = kein Trade
        if state.vol_regime == "high" and state.fractal_D > 1.7:
            logger.info("WAIT: Hohe Volatilitaet + hohe Fraktaldimension.")
            return TradingSignal(
                action="wait",
                entry_price=current_price,
                stop_loss=current_price,
                take_profit=current_price,
                position_size_pct=0.0,
                confidence=0.0,
                reason=f"Hohes Vol-Regime + fraktales Chaos (D={state.fractal_D:.3f})"
            )

        # --- Phase-Space-Regime berechnen ---
        from wbot.forecast.phase_space import get_phase_space_regime_from_prices
        phase_regime = get_phase_space_regime_from_prices(
            np.append(state.returns.cumsum(), 0) + current_price
            if len(state.returns) > 0 else np.array([current_price, current_price, current_price]),
            window=20
        )

        # --- 2. BREAKOUT-SIGNAL ---
        # Berechne Wahrscheinlichkeit Range > breakout_threshold_pct
        prob_large_range = _estimate_prob_range_above(forecast, breakout_threshold_pct)

        if (prob_large_range > breakout_prob_threshold and
                phase_regime in ["uptrend_attractor", "downtrend_attractor"]):

            # Richtung aus Phase-Space
            if phase_regime == "uptrend_attractor":
                action = "long_breakout"
                direction = "up"
                entry_price = current_price * (1 + 0.001)  # 0.1% Buy-Stop
                tp_price = _find_nearest_attractor(state.liquidity_attractors, entry_price, "up")
                # SL = Q95 der Low-Verteilung
                sl_offset_pct = forecast.q95_range_pct / 2.0 / 100.0
                stop_loss = entry_price * (1 - sl_offset_pct)
            else:
                action = "short_breakout"
                direction = "down"
                entry_price = current_price * (1 - 0.001)  # 0.1% Sell-Stop
                tp_price = _find_nearest_attractor(state.liquidity_attractors, entry_price, "down")
                sl_offset_pct = forecast.q95_range_pct / 2.0 / 100.0
                stop_loss = entry_price * (1 + sl_offset_pct)

            # Validiere: TP muss sinnvoll sein
            if action == "long_breakout" and tp_price <= entry_price:
                tp_price = entry_price * (1 + forecast.expected_high_pct / 100.0)
            if action == "short_breakout" and tp_price >= entry_price:
                tp_price = entry_price * (1 - forecast.expected_low_pct / 100.0)

            size_pct = _compute_position_size(
                risk_per_trade_pct, entry_price, stop_loss, leverage
            )

            # Confidence basierend auf Wahrscheinlichkeit und Regime-Staerke
            confidence = min(1.0, prob_large_range * 1.2)

            logger.info(
                f"SIGNAL: {action} | entry={entry_price:.2f} | "
                f"sl={stop_loss:.2f} | tp={tp_price:.2f} | "
                f"size={size_pct:.1f}% | conf={confidence:.2f}"
            )

            return TradingSignal(
                action=action,
                entry_price=round(entry_price, 8),
                stop_loss=round(stop_loss, 8),
                take_profit=round(tp_price, 8),
                position_size_pct=size_pct,
                confidence=confidence,
                reason=f"Breakout-Signal: P(Range>{breakout_threshold_pct}%)={prob_large_range:.2%}, "
                       f"Phase={phase_regime}"
            )

        # --- 3. RANGE-SIGNAL ---
        # Berechne Wahrscheinlichkeit Range < range_threshold_pct
        prob_small_range = _estimate_prob_range_below(forecast, range_threshold_pct)

        if (prob_small_range > range_prob_threshold and
                phase_regime == "range_zone"):

            # Long nahe erwartetem Low, TP = Mitte der Range
            # Entry = aktueller Preis - halber erwarteter Low-Abstand
            entry_offset = (forecast.expected_low_pct / 2.0) / 100.0
            entry_price = current_price * (1 - entry_offset)

            # TP = zurueck zur Mitte (aktueller Preis)
            tp_price = current_price

            # SL = Q95 der Range nach unten
            sl_offset_pct = forecast.q95_range_pct / 2.0 / 100.0
            stop_loss = current_price * (1 - sl_offset_pct)

            # Validierung: Entry muss unter TP liegen
            if entry_price >= tp_price:
                entry_price = current_price * 0.999

            size_pct = _compute_position_size(
                risk_per_trade_pct, entry_price, stop_loss, leverage
            )

            confidence = min(1.0, prob_small_range * 1.1)

            logger.info(
                f"SIGNAL: long_range | entry={entry_price:.2f} | "
                f"sl={stop_loss:.2f} | tp={tp_price:.2f} | "
                f"size={size_pct:.1f}% | conf={confidence:.2f}"
            )

            return TradingSignal(
                action="long_range",
                entry_price=round(entry_price, 8),
                stop_loss=round(stop_loss, 8),
                take_profit=round(tp_price, 8),
                position_size_pct=size_pct,
                confidence=confidence,
                reason=f"Range-Signal: P(Range<{range_threshold_pct}%)={prob_small_range:.2%}, "
                       f"Phase={phase_regime}"
            )

        # --- 4. KEIN SIGNAL ---
        logger.info(
            f"WAIT | P(large_range)={prob_large_range:.2%} < {breakout_prob_threshold} | "
            f"P(small_range)={prob_small_range:.2%} < {range_prob_threshold} | "
            f"phase={phase_regime}"
        )
        return TradingSignal(
            action="wait",
            entry_price=current_price,
            stop_loss=current_price,
            take_profit=current_price,
            position_size_pct=0.0,
            confidence=0.0,
            reason=(
                f"Kein Signal: P(large_range)={prob_large_range:.2%}, "
                f"P(small_range)={prob_small_range:.2%}, Phase={phase_regime}"
            )
        )

    except Exception as e:
        logger.error(f"Signal-Generierung fehlgeschlagen: {e}", exc_info=True)
        return wait_signal


def _estimate_prob_range_above(forecast, threshold_pct: float) -> float:
    """
    Schaetzt P(Range > threshold_pct) aus der Range-Forecast-Verteilung.

    Interpoliert zwischen bekannten Quantil-Punkten.
    """
    try:
        # Quantil-Punkte bekannt: q50, q75, q90, q95, E[range]
        if threshold_pct <= forecast.q50_range_pct:
            return 0.75  # > Median => ~50% + Interpolation
        elif threshold_pct <= forecast.q75_range_pct:
            # Interpoliere zwischen 50% und 25%
            t = (threshold_pct - forecast.q50_range_pct) / max(
                forecast.q75_range_pct - forecast.q50_range_pct, 0.01
            )
            return 0.50 - t * 0.25
        elif threshold_pct <= forecast.q90_range_pct:
            t = (threshold_pct - forecast.q75_range_pct) / max(
                forecast.q90_range_pct - forecast.q75_range_pct, 0.01
            )
            return 0.25 - t * 0.15
        elif threshold_pct <= forecast.q95_range_pct:
            t = (threshold_pct - forecast.q90_range_pct) / max(
                forecast.q95_range_pct - forecast.q90_range_pct, 0.01
            )
            return 0.10 - t * 0.05
        else:
            return 0.05  # Sehr grosses Threshold

    except Exception:
        return 0.0


def _estimate_prob_range_below(forecast, threshold_pct: float) -> float:
    """
    Schaetzt P(Range < threshold_pct) aus der Range-Forecast-Verteilung.
    """
    try:
        prob_above = _estimate_prob_range_above(forecast, threshold_pct)
        return max(0.0, 1.0 - prob_above)
    except Exception:
        return 0.0
