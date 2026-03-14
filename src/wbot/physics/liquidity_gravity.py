# src/wbot/physics/liquidity_gravity.py
# Modul 14-15: Liquiditaetsfeld + Gravitationskraft-Berechnung
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Markt-Gravitationskonstante
G = 0.01


def liquidity_density(
    price_series: np.ndarray,
    volume_series: np.ndarray,
    bins: int = 100
) -> tuple:
    """
    Berechnet die Liquiditaetsdichte rho_L(P) als gewichtetes Histogramm.

    Preislevel mit hohem Handelsvolumen = hohe Liquiditaet.

    Args:
        price_series: Array von Schlusskursen
        volume_series: Array von Handelsvolumen
        bins: Anzahl Preis-Bins

    Returns:
        (density_array, price_levels_array) - Liquiditaetsdichte und zugehoerige Preislevel
    """
    if len(price_series) < 2 or len(volume_series) < 2:
        logger.warning("Zu wenig Daten fuer Liquiditaetsdichte.")
        dummy_levels = np.linspace(0.0, 1.0, bins)
        return np.ones(bins) / bins, dummy_levels

    try:
        n = min(len(price_series), len(volume_series))
        prices = np.array(price_series[-n:], dtype=float)
        volumes = np.array(volume_series[-n:], dtype=float)

        p_min = np.min(prices)
        p_max = np.max(prices)

        if p_max - p_min < 1e-10:
            # Alle Preise identisch
            density = np.zeros(bins)
            density[bins // 2] = 1.0
            price_levels = np.linspace(p_min * 0.99, p_max * 1.01, bins)
            return density, price_levels

        bin_edges = np.linspace(p_min, p_max, bins + 1)
        price_levels = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Gewichtetes Histogramm: Volumen als Gewicht
        density = np.zeros(bins)
        bin_indices = np.digitize(prices, bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, bins - 1)

        for i, (idx, vol) in enumerate(zip(bin_indices, volumes)):
            density[idx] += vol

        # Normiere Dichte
        total = density.sum()
        if total > 0:
            density = density / total

        return density, price_levels

    except Exception as e:
        logger.warning(f"Liquiditaetsdichte Berechnung fehlgeschlagen: {e}")
        dummy_levels = np.linspace(float(np.min(price_series)), float(np.max(price_series)), bins)
        return np.ones(bins) / bins, dummy_levels


def gravity_potential(
    price: float,
    density: np.ndarray,
    price_levels: np.ndarray
) -> float:
    """
    Berechnet das Gravitationspotential U(P) = -sum(G * rho_i / |P - P_i|).

    Regionen hoher Liquiditaet ziehen den Preis an (negative Potential-Barriere).

    Args:
        price: Aktueller Preis
        density: Liquiditaetsdichte-Array
        price_levels: Zugehoerige Preislevel

    Returns:
        U(P) - Gravitationspotential
    """
    if len(density) == 0 or len(price_levels) == 0:
        return 0.0

    try:
        distances = np.abs(price - price_levels)
        # Vermeide Division durch Null: Mindestabstand
        distances = np.maximum(distances, price * 0.0001 if price > 0 else 0.0001)

        U = -G * np.sum(density / distances)
        return float(U)

    except Exception as e:
        logger.warning(f"Gravitationspotential Berechnung fehlgeschlagen: {e}")
        return 0.0


def gravity_force(
    price: float,
    density: np.ndarray,
    price_levels: np.ndarray,
    delta: float = 0.001
) -> float:
    """
    Berechnet die Gravitationskraft F_L = -dU/dP (numerischer Gradient).

    Positive Kraft = Preiszug nach oben (Attraktor oberhalb)
    Negative Kraft = Preiszug nach unten (Attraktor unterhalb)

    Args:
        price: Aktueller Preis
        density: Liquiditaetsdichte-Array
        price_levels: Zugehoerige Preislevel
        delta: Finite-Differenz-Schrittweite (relativ zum Preis)

    Returns:
        F_L - Gravitationskraft am aktuellen Preis
    """
    if len(density) == 0 or price <= 0:
        return 0.0

    try:
        dp = price * delta
        U_plus = gravity_potential(price + dp, density, price_levels)
        U_minus = gravity_potential(price - dp, density, price_levels)

        F = -(U_plus - U_minus) / (2 * dp)
        return float(F)

    except Exception as e:
        logger.warning(f"Gravitationskraft Berechnung fehlgeschlagen: {e}")
        return 0.0


def find_liquidity_attractors(
    density: np.ndarray,
    price_levels: np.ndarray,
    n_top: int = 5
) -> list:
    """
    Findet die Top-N Liquiditaetszonen (lokale Maxima der Dichtekurve).

    Args:
        density: Liquiditaetsdichte-Array
        price_levels: Zugehoerige Preislevel
        n_top: Anzahl der Top-Attraktoren

    Returns:
        Liste von (preis, staerke) Tupeln, sortiert nach Staerke (absteigend)
    """
    if len(density) < 3 or len(price_levels) < 3:
        return []

    try:
        # Finde lokale Maxima
        local_maxima = []
        for i in range(1, len(density) - 1):
            if density[i] > density[i - 1] and density[i] > density[i + 1] and density[i] > 0:
                local_maxima.append((float(price_levels[i]), float(density[i])))

        if not local_maxima:
            # Fallback: Nimm einfach die Top-N Bins
            top_indices = np.argsort(density)[-n_top:][::-1]
            local_maxima = [(float(price_levels[i]), float(density[i])) for i in top_indices if density[i] > 0]

        # Sortiere nach Staerke (Dichte) absteigend
        local_maxima.sort(key=lambda x: x[1], reverse=True)

        return local_maxima[:n_top]

    except Exception as e:
        logger.warning(f"Liquiditaets-Attraktoren Suche fehlgeschlagen: {e}")
        return []
