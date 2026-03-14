# src/wbot/physics/fractal_dimension.py
# Modul 7-9: Hurst-Exponent + Fraktaldimension + Regime-Klassifikation
import numpy as np
import logging

logger = logging.getLogger(__name__)


def hurst_exponent(prices: np.ndarray, min_lag: int = 2, max_lag: int = 50) -> float:
    """
    Berechnet den Hurst-Exponenten via R/S-Analyse (Rescaled Range Analysis).

    H < 0.45: Mean-reverting (anti-persistentes Verhalten)
    H ~ 0.5:  Random Walk (kein Gedaechtnis)
    H > 0.55: Trending (persistentes Verhalten)

    Args:
        prices: Array von Preisen
        min_lag: Minimale Lag-Laenge fuer R/S
        max_lag: Maximale Lag-Laenge

    Returns:
        Hurst-Exponent H (0 < H < 1)
    """
    if len(prices) < max_lag + 5:
        logger.warning(f"Zu wenig Daten fuer Hurst-Exponent ({len(prices)} < {max_lag + 5}), nutze 0.5.")
        return 0.5

    try:
        # Log-Returns
        log_returns = np.log(prices[1:] / prices[:-1])

        lags = range(min_lag, min(max_lag, len(log_returns) // 2))
        rs_values = []
        lag_values = []

        for lag in lags:
            n_segments = len(log_returns) // lag
            if n_segments < 1:
                continue

            rs_list = []
            for seg in range(n_segments):
                segment = log_returns[seg * lag:(seg + 1) * lag]
                if len(segment) < 2:
                    continue

                mean_seg = np.mean(segment)
                deviated = segment - mean_seg
                cumulative = np.cumsum(deviated)

                r = np.max(cumulative) - np.min(cumulative)  # Range
                s = np.std(segment, ddof=1)  # Standardabweichung

                if s > 0 and r > 0:
                    rs_list.append(r / s)

            if rs_list:
                rs_values.append(np.log(np.mean(rs_list)))
                lag_values.append(np.log(lag))

        if len(rs_values) < 3:
            logger.warning("Nicht genug R/S-Werte fuer Regression, nutze H=0.5.")
            return 0.5

        # Lineare Regression: log(R/S) = H * log(lag) + const
        lag_arr = np.array(lag_values)
        rs_arr = np.array(rs_values)

        coeffs = np.polyfit(lag_arr, rs_arr, 1)
        H = float(coeffs[0])

        # H auf plausiblen Bereich beschraenken
        H = max(0.01, min(0.99, H))
        logger.debug(f"Hurst-Exponent H={H:.4f}")
        return H

    except Exception as e:
        logger.warning(f"Hurst-Exponent Berechnung fehlgeschlagen: {e}. Nutze H=0.5.")
        return 0.5


def fractal_dimension(prices: np.ndarray) -> float:
    """
    Berechnet die Fraktaldimension als D = 2 - H.

    D nahe 1.0: Stark trending (glatte Kurve)
    D nahe 1.5: Random Walk
    D nahe 2.0: Sehr volatil / Chaos (flaechen-fuellend)

    Args:
        prices: Array von Preisen

    Returns:
        Fraktaldimension D (1 < D < 2)
    """
    try:
        H = hurst_exponent(prices)
        D = 2.0 - H
        D = max(1.01, min(1.99, D))
        logger.debug(f"Fraktaldimension D={D:.4f}")
        return D
    except Exception as e:
        logger.warning(f"Fraktaldimension Berechnung fehlgeschlagen: {e}. Nutze D=1.5.")
        return 1.5


def fractal_regime(D: float) -> str:
    """
    Klassifiziert das Markt-Regime basierend auf der Fraktaldimension.

    Args:
        D: Fraktaldimension (1 < D < 2)

    Returns:
        "trend" | "random" | "chaos"
    """
    try:
        if D < 1.45:
            return "trend"
        elif D < 1.55:
            return "random"
        else:
            return "chaos"
    except Exception:
        return "random"
