# src/wbot/physics/market_energy.py
# Modul 16: Marktenergie - multi-timeframe Energieberechnung
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def candle_energy(prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """
    Berechnet die Kerzenenergie E_t = V_t * (Delta_P_t)^2.

    Analogie zur kinetischen Energie: Masse (Volumen) * Geschwindigkeit^2 (Preisaenderung)

    Args:
        prices: Array von Schlusskursen
        volumes: Array von Handelsvolumen

    Returns:
        Array von Energiewerten E_t (Laenge = len(prices) - 1)
    """
    if len(prices) < 2 or len(volumes) < 2:
        return np.array([0.0])

    try:
        n = min(len(prices), len(volumes))
        prices = np.array(prices[:n], dtype=float)
        volumes = np.array(volumes[:n], dtype=float)

        # Log-Returns als Delta_P
        delta_p = np.diff(np.log(prices + 1e-10))

        # E_t = V_t * (delta_P_t)^2
        vol_slice = volumes[1:]  # Volumen der Folge-Kerze
        energy = vol_slice * delta_p ** 2

        return energy

    except Exception as e:
        logger.warning(f"Kerzenenergie Berechnung fehlgeschlagen: {e}")
        return np.array([0.0])


def multi_timeframe_energy(
    ohlcv_df: pd.DataFrame,
    windows: list = None
) -> float:
    """
    Berechnet die Multi-Timeframe-Marktenergie als gewichtete Summe.

    E_total = sum(w_i * E_i) mit w_i = 1/window_i (kuerzere Fenster = mehr Gewicht)

    Args:
        ohlcv_df: DataFrame mit Spalten [open, high, low, close, volume]
        windows: Liste von Fensterlaengen [5, 14, 30]

    Returns:
        E_total (skalierte Gesamtenergie)
    """
    if windows is None:
        windows = [5, 14, 30]

    if ohlcv_df is None or len(ohlcv_df) < max(windows) + 2:
        logger.warning("Zu wenig Daten fuer Multi-Timeframe-Energie.")
        return 0.0

    try:
        prices = ohlcv_df['close'].values.astype(float)
        volumes = ohlcv_df['volume'].values.astype(float)

        all_energy = candle_energy(prices, volumes)

        if len(all_energy) == 0:
            return 0.0

        total_energy = 0.0
        total_weight = 0.0

        for window in windows:
            w_i = 1.0 / window
            # Mittlere Energie ueber das Fenster
            window_energy = all_energy[-window:] if len(all_energy) >= window else all_energy
            if len(window_energy) == 0:
                continue

            E_i = float(np.mean(window_energy))
            total_energy += w_i * E_i
            total_weight += w_i

        if total_weight > 0:
            E_total = total_energy / total_weight
        else:
            E_total = 0.0

        logger.debug(f"Multi-Timeframe-Energie E_total={E_total:.8f}")
        return E_total

    except Exception as e:
        logger.warning(f"Multi-Timeframe-Energie Berechnung fehlgeschlagen: {e}")
        return 0.0


def energy_regime(
    E_total: float,
    threshold_high: float = 0.0001,
    threshold_low: float = 0.00001
) -> str:
    """
    Klassifiziert das Energie-Regime.

    Args:
        E_total: Gesamtenergie
        threshold_high: Schwellenwert fuer hohes Regime
        threshold_low: Schwellenwert fuer niedriges Regime

    Returns:
        "high" | "normal" | "low"
    """
    try:
        if E_total >= threshold_high:
            return "high"
        elif E_total <= threshold_low:
            return "low"
        else:
            return "normal"
    except Exception:
        return "normal"
