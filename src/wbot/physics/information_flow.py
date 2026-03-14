# src/wbot/physics/information_flow.py
# Modul 12-13: Informationsfluss + Shannon-Entropie + Transfer-Entropie
import numpy as np
import logging

logger = logging.getLogger(__name__)


def entropy(returns: np.ndarray, bins: int = 20) -> float:
    """
    Berechnet die Shannon-Entropie der Returns-Verteilung.

    I = -sum(p_i * log(p_i))

    Args:
        returns: Array von Log-Returns
        bins: Anzahl Histogramm-Bins

    Returns:
        Shannon-Entropie I (>= 0)
    """
    if len(returns) < 5:
        return 0.0

    try:
        counts, _ = np.histogram(returns, bins=bins)
        probabilities = counts / counts.sum()

        # Nur nicht-null Wahrscheinlichkeiten beruecksichtigen
        probabilities = probabilities[probabilities > 0]

        if len(probabilities) == 0:
            return 0.0

        H = -float(np.sum(probabilities * np.log(probabilities)))
        return max(0.0, H)

    except Exception as e:
        logger.warning(f"Entropie-Berechnung fehlgeschlagen: {e}")
        return 0.0


def information_flow(returns: np.ndarray, window: int = 20) -> float:
    """
    Berechnet den Informationsfluss Phi_t = dI/dt als differenzierte Entropie-Zeitreihe.

    Ein hoher Phi_t deutet auf groessere erwartete Range hin.

    Args:
        returns: Array von Log-Returns
        window: Fensterlaenge fuer rollende Entropie

    Returns:
        Phi_t (aktueller Informationsfluss)
    """
    if len(returns) < window + 5:
        return 0.0

    try:
        # Berechne rollende Entropie ueber die letzten Fenster
        n_steps = min(5, len(returns) - window)  # Letzte 5 Fenster-Positionen
        entropy_series = []

        for i in range(n_steps):
            start = len(returns) - window - (n_steps - 1 - i)
            end = start + window
            if start < 0 or end > len(returns):
                continue
            h = entropy(returns[start:end])
            entropy_series.append(h)

        if len(entropy_series) < 2:
            return 0.0

        # Ableitung: Differenz der letzten zwei Entropiewerte
        phi = entropy_series[-1] - entropy_series[-2]
        return float(phi)

    except Exception as e:
        logger.warning(f"Informationsfluss-Berechnung fehlgeschlagen: {e}")
        return 0.0


def transfer_entropy(x: np.ndarray, y: np.ndarray, lag: int = 1, bins: int = 10) -> float:
    """
    Berechnet die Transfer-Entropie T(X->Y) = Informationsuebergang von X nach Y.

    T(X->Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-lag})

    Args:
        x: Quell-Zeitreihe (z.B. Volume-Changes)
        y: Ziel-Zeitreihe (z.B. Price-Returns)
        lag: Zeitverzoegerung
        bins: Anzahl Diskretisierungs-Bins

    Returns:
        Transfer-Entropie T(X->Y) >= 0
    """
    if len(x) < lag + bins or len(y) < lag + bins:
        return 0.0

    try:
        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]

        # Diskretisierung durch Binning
        def discretize(arr, n_bins):
            bin_edges = np.linspace(np.min(arr) - 1e-10, np.max(arr) + 1e-10, n_bins + 1)
            return np.digitize(arr, bin_edges) - 1

        x_disc = discretize(x, bins)
        y_disc = discretize(y, bins)

        # Zeitreihen verschieben
        y_t = y_disc[lag:]
        y_t1 = y_disc[:-lag]
        x_t1 = x_disc[:-lag]

        n_valid = len(y_t)
        if n_valid < 10:
            return 0.0

        # Berechne Joint- und Marginal-Entropien
        def joint_prob(*arrays):
            """Berechne gemeinsame Wahrscheinlichkeitsverteilung."""
            coords = np.array(arrays).T
            unique, counts = np.unique(coords, axis=0, return_counts=True)
            probs = counts / counts.sum()
            return probs

        def joint_entropy(*arrays):
            probs = joint_prob(*arrays)
            probs = probs[probs > 0]
            return -np.sum(probs * np.log(probs + 1e-15))

        # H(Y_t | Y_{t-1}) = H(Y_t, Y_{t-1}) - H(Y_{t-1})
        h_yt_yt1 = joint_entropy(y_t, y_t1) - joint_entropy(y_t1)

        # H(Y_t | Y_{t-1}, X_{t-lag}) = H(Y_t, Y_{t-1}, X_{t-1}) - H(Y_{t-1}, X_{t-1})
        h_yt_yt1_xt1 = joint_entropy(y_t, y_t1, x_t1) - joint_entropy(y_t1, x_t1)

        te = h_yt_yt1 - h_yt_yt1_xt1
        return float(max(0.0, te))

    except Exception as e:
        logger.warning(f"Transfer-Entropie Berechnung fehlgeschlagen: {e}")
        return 0.0
