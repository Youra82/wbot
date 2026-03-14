# src/wbot/forecast/phase_space.py
# 4D Phasenraum-Analyse: P, v, a, rho_L (Preis, Geschwindigkeit, Beschleunigung, Dichte)
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def compute_phase_space_vector(prices: np.ndarray, window: int = 20) -> Tuple[float, float, float, float]:
    """
    Berechnet den 4D Phasenraum-Vektor (P, v, a, rho_proxy) aus Preisdaten.

    - P: Aktueller Preis
    - v: dP/dt (Geschwindigkeit = letzter 1-Kerzen-Return)
    - a: d^2P/dt^2 (Beschleunigung = Aenderung von v)
    - rho_proxy: Lokale Volumen-Dichte (normiert, Proxy fuer Marktdichte)

    Args:
        prices: Array von Preisen (mindestens 3 Werte)
        window: Fenster fuer Rho-Proxy Normierung

    Returns:
        (P, v, a, rho_proxy) Tupel
    """
    if len(prices) < 3:
        logger.warning("Zu wenig Daten fuer Phasenraum-Vektor.")
        p = float(prices[-1]) if len(prices) > 0 else 0.0
        return p, 0.0, 0.0, 0.0

    try:
        # Aktueller Preis
        P = float(prices[-1])

        # Geschwindigkeit: letzter Log-Return
        v = float(np.log(prices[-1] / prices[-2]))

        # Beschleunigung: Aenderung des Log-Returns
        v_prev = float(np.log(prices[-2] / prices[-3]))
        a = v - v_prev

        # Rho-Proxy: normierte Preisvolatilitaet als Dichtemass
        recent = prices[-window:] if len(prices) >= window else prices
        if len(recent) >= 2:
            local_vol = float(np.std(np.diff(np.log(recent + 1e-10))))
            # Normiere auf [0, 1] mit Sigmoid-Transformation
            rho_proxy = float(1.0 / (1.0 + np.exp(-local_vol * 50)))
        else:
            rho_proxy = 0.5

        logger.debug(f"Phasenraum: P={P:.2f}, v={v:.6f}, a={a:.6f}, rho={rho_proxy:.4f}")
        return P, v, a, rho_proxy

    except Exception as e:
        logger.warning(f"Phasenraum-Vektor Berechnung fehlgeschlagen: {e}")
        p = float(prices[-1]) if len(prices) > 0 else 0.0
        return p, 0.0, 0.0, 0.0


def classify_phase_space_regime(v: float, a: float, rho: float) -> str:
    """
    Klassifiziert das Phasenraum-Regime aus (v, a, rho).

    Entspricht den 4 Zonen:
    1. uptrend_attractor:    Preis steigt, positiver Schwung
    2. downtrend_attractor:  Preis faellt, negativer Schwung
    3. range_zone:           Kleine Bewegungen, keine klare Richtung
    4. crash_spike_zone:     Extreme Beschleunigung (potentielle Umkehr)

    Args:
        v: Preisgeschwindigkeit (Log-Return letzter Kerze)
        a: Beschleunigung (Aenderung von v)
        rho: Lokale Dichte-Proxy (0-1)

    Returns:
        Regime-Name als String
    """
    try:
        abs_v = abs(v)
        abs_a = abs(a)

        # Schwellenwerte (als Log-Return-Groessen)
        STRONG_MOVE_THRESHOLD = 0.005   # 0.5% pro Kerze
        MODERATE_MOVE = 0.002           # 0.2% pro Kerze
        HIGH_ACCEL = 0.003              # starke Beschleunigung
        HIGH_DENSITY = 0.65             # hohe Marktdichte

        # Crash/Spike-Zone: Extreme Beschleunigung + Richtungswechsel
        if abs_a > HIGH_ACCEL and abs_v > STRONG_MOVE_THRESHOLD:
            return "crash_spike_zone"

        # Uptrend-Attraktor: positiver Schwung + zunehmende oder stabile Bewegung
        if v > MODERATE_MOVE and a >= -MODERATE_MOVE:
            return "uptrend_attractor"

        # Downtrend-Attraktor: negativer Schwung + zunehmende oder stabile Bewegung
        if v < -MODERATE_MOVE and a <= MODERATE_MOVE:
            return "downtrend_attractor"

        # Range-Zone: kleine Bewegungen
        return "range_zone"

    except Exception as e:
        logger.warning(f"Phasenraum-Regime Klassifikation fehlgeschlagen: {e}")
        return "range_zone"


def get_phase_space_regime_from_prices(prices: np.ndarray, window: int = 20) -> str:
    """
    Convenience-Funktion: Berechnet Phasenraum und klassifiziert Regime.

    Args:
        prices: Array von Preisen
        window: Fenster fuer Rho-Proxy

    Returns:
        Regime-Name
    """
    try:
        P, v, a, rho = compute_phase_space_vector(prices, window=window)
        regime = classify_phase_space_regime(v, a, rho)
        logger.info(f"Phasenraum-Regime: {regime} (v={v:.5f}, a={a:.5f}, rho={rho:.3f})")
        return regime
    except Exception as e:
        logger.warning(f"Phasenraum-Regime Berechnung fehlgeschlagen: {e}")
        return "range_zone"
