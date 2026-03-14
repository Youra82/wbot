# src/wbot/forecast/range_forecast.py
# Modul 20: Range-Wahrscheinlichkeitsverteilung aus Monte-Carlo-Ergebnissen
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RangeForecast:
    """Vollstaendige Prognose fuer Range und Kerzenform der naechsten Kerze."""
    # Range-Statistiken
    expected_range_pct: float      # Erwartete Range in %
    q50_range_pct: float           # Median-Range
    q75_range_pct: float           # 75%-Quantil
    q90_range_pct: float           # 90%-Quantil
    q95_range_pct: float           # 95%-Quantil

    # Wahrscheinlichkeiten fuer High-Brueche
    prob_high_1pct: float          # P(High > +1%)
    prob_high_2pct: float          # P(High > +2%)
    prob_high_3pct: float          # P(High > +3%)

    # Wahrscheinlichkeiten fuer Low-Unterschreitungen
    prob_low_1pct: float           # P(Low < -1%)
    prob_low_2pct: float           # P(Low < -2%)
    prob_low_3pct: float           # P(Low < -3%)

    # Kerzenform-Prognosen
    prob_bullish: float            # P(Close > Open)
    prob_bearish: float            # P(Close < Open)
    prob_doji: float               # P(body < 20% range)
    prob_pinbar: float             # P(wick > 60% range)

    # Erwartete Extremwerte
    expected_high_pct: float       # E[High] in % ueber current price
    expected_low_pct: float        # E[Low] in % unter current price (positiver Wert = nach unten)


def compute_range_forecast(sim, current_price: float) -> RangeForecast:
    """
    Berechnet die Range-Prognose aus Monte-Carlo-Simulationsergebnissen.

    Args:
        sim: SimulationResult Objekt
        current_price: Aktueller Preis (Open der naechsten Kerze)

    Returns:
        RangeForecast mit allen Wahrscheinlichkeiten
    """
    n = len(sim.highs)
    if n == 0:
        logger.error("Leere Simulation - kann keine Prognose erstellen.")
        return _fallback_forecast(current_price)

    try:
        price = current_price if current_price > 0 else 1.0

        # --- Range-Statistiken ---
        ranges = sim.ranges  # bereits in % (high-low)/price*100
        expected_range = float(np.mean(ranges))
        q50 = float(np.percentile(ranges, 50))
        q75 = float(np.percentile(ranges, 75))
        q90 = float(np.percentile(ranges, 90))
        q95 = float(np.percentile(ranges, 95))

        # --- High-Wahrscheinlichkeiten ---
        high_pct = (sim.highs - price) / price * 100.0
        prob_h1 = float(np.mean(high_pct > 1.0))
        prob_h2 = float(np.mean(high_pct > 2.0))
        prob_h3 = float(np.mean(high_pct > 3.0))

        # --- Low-Wahrscheinlichkeiten ---
        low_pct = (price - sim.lows) / price * 100.0  # positiv wenn Low unter price
        prob_l1 = float(np.mean(low_pct > 1.0))
        prob_l2 = float(np.mean(low_pct > 2.0))
        prob_l3 = float(np.mean(low_pct > 3.0))

        # --- Kerzenform-Wahrscheinlichkeiten ---
        prob_bullish = float(np.mean(sim.closes > price))   # Close > Open (=price)
        prob_bearish = float(np.mean(sim.closes < price))
        prob_doji = float(np.mean(sim.body_sizes < 0.20))   # Koerper < 20% Range
        prob_pinbar = float(np.mean(                          # groesster Docht > 60% Range
            np.maximum(sim.upper_wicks, sim.lower_wicks) > 0.60
        ))

        # --- Erwartete Extremwerte ---
        expected_high_pct = float(np.mean(high_pct))
        expected_low_pct = float(np.mean(low_pct))

        forecast = RangeForecast(
            expected_range_pct=expected_range,
            q50_range_pct=q50,
            q75_range_pct=q75,
            q90_range_pct=q90,
            q95_range_pct=q95,
            prob_high_1pct=prob_h1,
            prob_high_2pct=prob_h2,
            prob_high_3pct=prob_h3,
            prob_low_1pct=prob_l1,
            prob_low_2pct=prob_l2,
            prob_low_3pct=prob_l3,
            prob_bullish=prob_bullish,
            prob_bearish=prob_bearish,
            prob_doji=prob_doji,
            prob_pinbar=prob_pinbar,
            expected_high_pct=expected_high_pct,
            expected_low_pct=expected_low_pct,
        )

        logger.info(
            f"RangeForecast | E[Range]={expected_range:.2f}% | "
            f"P(Bull)={prob_bullish:.2%} | P(H>1%)={prob_h1:.2%} | P(L>1%)={prob_l1:.2%}"
        )
        return forecast

    except Exception as e:
        logger.error(f"Range-Prognose fehlgeschlagen: {e}", exc_info=True)
        return _fallback_forecast(current_price)


def _fallback_forecast(current_price: float) -> RangeForecast:
    """Gibt eine neutrale Fallback-Prognose zurueck."""
    return RangeForecast(
        expected_range_pct=2.0,
        q50_range_pct=1.8,
        q75_range_pct=2.5,
        q90_range_pct=3.5,
        q95_range_pct=4.5,
        prob_high_1pct=0.5,
        prob_high_2pct=0.3,
        prob_high_3pct=0.15,
        prob_low_1pct=0.5,
        prob_low_2pct=0.3,
        prob_low_3pct=0.15,
        prob_bullish=0.5,
        prob_bearish=0.5,
        prob_doji=0.15,
        prob_pinbar=0.1,
        expected_high_pct=1.0,
        expected_low_pct=1.0,
    )


def format_forecast_table(forecast: RangeForecast) -> str:
    """
    Formatiert die Range-Prognose als schoene ASCII-Tabelle fuer das Logging.

    Args:
        forecast: RangeForecast Objekt

    Returns:
        Formatierter ASCII-String
    """
    lines = [
        "",
        "=" * 56,
        "     wbot QGRS - Range Forecast (naechste Kerze)",
        "=" * 56,
        "",
        "  RANGE-STATISTIKEN:",
        f"  E[Range]:         {forecast.expected_range_pct:>7.2f}%",
        f"  Median-Range:     {forecast.q50_range_pct:>7.2f}%",
        f"  75%-Quantil:      {forecast.q75_range_pct:>7.2f}%",
        f"  90%-Quantil:      {forecast.q90_range_pct:>7.2f}%",
        f"  95%-Quantil:      {forecast.q95_range_pct:>7.2f}%",
        "",
        "  PREISZIEL-WAHRSCHEINLICHKEITEN:",
        f"  P(High > +1%):    {forecast.prob_high_1pct:>7.1%}",
        f"  P(High > +2%):    {forecast.prob_high_2pct:>7.1%}",
        f"  P(High > +3%):    {forecast.prob_high_3pct:>7.1%}",
        f"  P(Low  < -1%):    {forecast.prob_low_1pct:>7.1%}",
        f"  P(Low  < -2%):    {forecast.prob_low_2pct:>7.1%}",
        f"  P(Low  < -3%):    {forecast.prob_low_3pct:>7.1%}",
        "",
        "  KERZENFORM-PROGNOSE:",
        f"  P(Bullish):       {forecast.prob_bullish:>7.1%}",
        f"  P(Bearish):       {forecast.prob_bearish:>7.1%}",
        f"  P(Doji):          {forecast.prob_doji:>7.1%}",
        f"  P(Pinbar):        {forecast.prob_pinbar:>7.1%}",
        "",
        "  ERWARTETE EXTREMWERTE:",
        f"  E[High]:          +{forecast.expected_high_pct:>6.2f}%",
        f"  E[Low]:           -{forecast.expected_low_pct:>6.2f}%",
        "",
        "=" * 56,
        "",
    ]
    return "\n".join(lines)
