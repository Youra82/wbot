# src/wbot/physics/chaos_indicators.py
# Modul 10-11: Lyapunov-Exponent + Attraktor-Regime-Klassifikation
import numpy as np
import logging

logger = logging.getLogger(__name__)


def phase_space_embedding(prices: np.ndarray, dim: int = 3, lag: int = 1) -> np.ndarray:
    """
    Erstellt eine Phasenraum-Einbettung der Zeitreihe (Takens-Theorem).

    Args:
        prices: Array von Preisen
        dim: Einbettungsdimension
        lag: Zeit-Verzoegerung

    Returns:
        2D numpy-Array der Form (n_points, dim)
    """
    n = len(prices)
    n_embedded = n - (dim - 1) * lag
    if n_embedded <= 0:
        logger.warning("Nicht genug Daten fuer Phasenraum-Einbettung.")
        return np.array([]).reshape(0, dim)

    embedded = np.zeros((n_embedded, dim))
    for i in range(n_embedded):
        for j in range(dim):
            embedded[i, j] = prices[i + j * lag]

    return embedded


def lyapunov_exponent(
    prices: np.ndarray,
    embedding_dim: int = 3,
    lag: int = 1,
    min_neighbors: int = 5
) -> float:
    """
    Approximiert den groessten Lyapunov-Exponenten via Rosenstein-Algorithmus (vereinfacht).

    Positiver Wert = chaotisches System
    Negativer / Null Wert = stabiles System

    Args:
        prices: Array von Preisen (mindestens 50 Werte empfohlen)
        embedding_dim: Phasenraum-Einbettungsdimension
        lag: Zeit-Verzoegerung fuer Einbettung
        min_neighbors: Mindestanzahl naechster Nachbarn

    Returns:
        lambda_approx (Lyapunov-Exponent Approximation)
    """
    if len(prices) < 50:
        logger.warning(f"Zu wenig Daten fuer Lyapunov-Exponent ({len(prices)} < 50), nutze 0.0.")
        return 0.0

    try:
        # Normiere Preise auf [0, 1] fuer numerische Stabilitaet
        p_min, p_max = np.min(prices), np.max(prices)
        if p_max - p_min < 1e-10:
            return 0.0
        p_norm = (prices - p_min) / (p_max - p_min)

        # Phasenraum-Einbettung
        embedded = phase_space_embedding(p_norm, dim=embedding_dim, lag=lag)
        n_points = len(embedded)

        if n_points < min_neighbors + 2:
            logger.warning("Zu wenig eingebettete Punkte fuer Lyapunov, nutze 0.0.")
            return 0.0

        # Mindestabstand-Parameter: Vermeide temporaere Korrelationen
        min_temporal_separation = max(1, embedding_dim * lag)

        divergences = []

        # Fuer jeden Punkt: finde den naechsten Nachbarn (ohne temporaere Nachbarn)
        # Vereinfachung: Nutze nur einen Teil der Punkte fuer Geschwindigkeit
        step = max(1, n_points // 100)  # Max. 100 Referenzpunkte
        ref_indices = range(0, n_points - min_temporal_separation - 1, step)

        for i in ref_indices:
            distances = np.linalg.norm(embedded - embedded[i], axis=1)
            # Maskiere temporaere Nachbarn
            mask = np.abs(np.arange(n_points) - i) > min_temporal_separation
            masked_dist = np.where(mask, distances, np.inf)

            # Finde naechsten Nachbarn
            nn_idx = int(np.argmin(masked_dist))
            d0 = masked_dist[nn_idx]

            if d0 < 1e-10 or np.isinf(d0):
                continue

            # Verfolge Divergenz nach einem Schritt
            i_next = min(i + 1, n_points - 1)
            nn_next = min(nn_idx + 1, n_points - 1)

            d1 = np.linalg.norm(embedded[i_next] - embedded[nn_next])

            if d1 > 1e-10:
                divergences.append(np.log(d1 / d0))

        if not divergences:
            return 0.0

        lambda_approx = float(np.mean(divergences))
        logger.debug(f"Lyapunov-Exponent lambda={lambda_approx:.4f}")
        return lambda_approx

    except Exception as e:
        logger.warning(f"Lyapunov-Exponent Berechnung fehlgeschlagen: {e}. Nutze 0.0.")
        return 0.0


def attractor_regime(lambda_val: float, fractal_D: float) -> str:
    """
    Klassifiziert das Attraktor-Regime basierend auf Lyapunov-Exponent und Fraktaldimension.

    Args:
        lambda_val: Lyapunov-Exponent
        fractal_D: Fraktaldimension D

    Returns:
        "trend" | "range" | "chaotic" | "crash_risk"
    """
    try:
        # Crash-Risiko: stark chaotisch + hohe Dimension
        if lambda_val > 0.3 and fractal_D > 1.7:
            return "crash_risk"

        # Chaotisch: positiver Lyapunov
        if lambda_val > 0.1:
            return "chaotic"

        # Trending: negativer / kleiner Lyapunov + niedrige Dimension
        if lambda_val < 0.0 and fractal_D < 1.5:
            return "trend"

        # Range: kleiner Lyapunov + mittlere/hohe Dimension
        return "range"

    except Exception:
        return "range"
