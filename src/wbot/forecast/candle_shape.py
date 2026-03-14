# src/wbot/forecast/candle_shape.py
# CSF-Model: Kerzenform-Prognose basierend auf Monte-Carlo-Statistiken
import numpy as np
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


def classify_candle_shape(body: float, upper_wick: float, lower_wick: float) -> str:
    """
    Klassifiziert die Kerzenform basierend auf normalisierten Verhaeltnissen.

    Alle Eingaben sind als Bruchteil der Gesamtrange (0-1) angegeben.

    Args:
        body: Koerpergroesse als Anteil der Range (|close-open| / range)
        upper_wick: Oberer Docht als Anteil der Range
        lower_wick: Unterer Docht als Anteil der Range

    Returns:
        Kerzenform-Klassifikation
    """
    try:
        body = float(np.clip(body, 0.0, 1.0))
        upper_wick = float(np.clip(upper_wick, 0.0, 1.0))
        lower_wick = float(np.clip(lower_wick, 0.0, 1.0))

        # Doji: sehr kleiner Koerper
        if body < 0.20:
            return "doji"

        max_wick = max(upper_wick, lower_wick)

        # Pinbar: einer der Doechte dominiert stark
        if max_wick > 0.60:
            if lower_wick > upper_wick:
                return "bullish_pinbar"   # Hammer / Bullish Reversal
            else:
                return "bearish_pinbar"   # Shooting Star / Bearish Reversal

        # Trend-Kerzen: groesser Koerper
        if body > 0.50:
            if lower_wick >= upper_wick:
                return "bullish_trend"    # Bullische Momentum-Kerze
            else:
                return "bearish_trend"    # Bärische Momentum-Kerze

        return "neutral"

    except Exception as e:
        logger.warning(f"Kerzenform-Klassifikation fehlgeschlagen: {e}")
        return "neutral"


def candle_shape_distribution(sim) -> Dict[str, float]:
    """
    Berechnet die Wahrscheinlichkeitsverteilung aller Kerzenformen aus der Simulation.

    Args:
        sim: SimulationResult Objekt

    Returns:
        Dictionary mit Kerzenform -> Wahrscheinlichkeit
    """
    shapes = [
        "bullish_trend", "bearish_trend", "doji",
        "bullish_pinbar", "bearish_pinbar", "neutral"
    ]
    counts = {s: 0 for s in shapes}

    if len(sim.body_sizes) == 0:
        return {s: 1.0 / len(shapes) for s in shapes}

    try:
        n = len(sim.body_sizes)
        for i in range(n):
            shape = classify_candle_shape(
                sim.body_sizes[i],
                sim.upper_wicks[i],
                sim.lower_wicks[i]
            )
            if shape in counts:
                counts[shape] += 1
            else:
                counts["neutral"] += 1

        # Normiere zu Wahrscheinlichkeiten
        distribution = {s: counts[s] / n for s in shapes}

        logger.debug(
            f"Kerzenform-Verteilung: "
            f"bull={distribution['bullish_trend']:.2%} | "
            f"bear={distribution['bearish_trend']:.2%} | "
            f"doji={distribution['doji']:.2%} | "
            f"pin_bull={distribution['bullish_pinbar']:.2%} | "
            f"pin_bear={distribution['bearish_pinbar']:.2%}"
        )
        return distribution

    except Exception as e:
        logger.warning(f"Kerzenform-Verteilung fehlgeschlagen: {e}")
        return {s: 1.0 / len(shapes) for s in shapes}


def most_likely_shape(distribution: Dict[str, float]) -> Tuple[str, float]:
    """
    Gibt die wahrscheinlichste Kerzenform zurueck.

    Args:
        distribution: Dictionary aus candle_shape_distribution()

    Returns:
        (shape_name, probability) Tuple
    """
    if not distribution:
        return "neutral", 0.0

    try:
        best_shape = max(distribution, key=lambda k: distribution[k])
        return best_shape, distribution[best_shape]
    except Exception as e:
        logger.warning(f"Most-Likely-Shape Bestimmung fehlgeschlagen: {e}")
        return "neutral", 0.0
