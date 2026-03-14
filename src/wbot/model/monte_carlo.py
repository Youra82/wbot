# src/wbot/model/monte_carlo.py
# Modul 19: Monte-Carlo Simulation Engine fuer Range-Prognose
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Ergebnis der Monte-Carlo-Simulation."""
    highs: np.ndarray           # (n_simulations,) - max jedes Pfades
    lows: np.ndarray            # (n_simulations,) - min jedes Pfades
    closes: np.ndarray          # (n_simulations,) - Endpreis jedes Pfades
    ranges: np.ndarray          # (n_simulations,) - (high-low) in %
    body_sizes: np.ndarray      # (n_simulations,) - |close-open| / range
    upper_wicks: np.ndarray     # (n_simulations,) - upper wick / range
    lower_wicks: np.ndarray     # (n_simulations,) - lower wick / range


def run_monte_carlo(
    state,
    n_simulations: int = 10000,
    n_steps: int = 288
) -> SimulationResult:
    """
    Fuehrt die Monte-Carlo-Simulation der Tageskerzen-Preispfade durch.

    Verwendet numpy Vektorisierung fuer maximale Geschwindigkeit.
    Beruecksichtigt Liquiditaets-Attraktoren als zusaetzliche Drift-Komponente.

    Args:
        state: MarketState Objekt mit allen Physics-Features
        n_simulations: Anzahl simulierter Pfade (default 10000)
        n_steps: Intraday-Schritte (default 288 = 5-Min-Schritte pro Tag)

    Returns:
        SimulationResult mit statistischen Groessen aus allen Pfaden
    """
    from wbot.model.price_process import compute_drift, compute_sigma

    logger.info(f"Starte {n_simulations} Monte-Carlo-Simulationen mit {n_steps} Schritten...")

    try:
        price = state.price
        mu = compute_drift(state)
        sigma = compute_sigma(state)
        dt = 1.0 / n_steps

        logger.info(f"GBM Parameter: mu={mu:.6f}, sigma={sigma:.6f}, dt={dt:.6f}")

        # --- Batch-Simulation (vektorisiert) ---
        # Z: (n_steps, n_simulations) - Standard-Normalverteilung
        Z = np.random.standard_normal((n_steps, n_simulations))

        drift_term = (mu - 0.5 * sigma ** 2) * dt
        diffusion_term = sigma * np.sqrt(dt) * Z

        # Kumulierte Returns: (n_steps, n_simulations)
        log_returns_cumsum = np.cumsum(drift_term + diffusion_term, axis=0)

        # Pfade: (n_steps + 1, n_simulations) - Zeile 0 = Startpreis
        paths = np.empty((n_steps + 1, n_simulations))
        paths[0, :] = price
        paths[1:, :] = price * np.exp(log_returns_cumsum)

        # --- Liquiditaets-Attraktor-Korrektur ---
        # Jeder Pfad wird leicht zum naechsten Attraktor gezogen
        if state.liquidity_attractors:
            _apply_gravity_correction(paths, state, mu, sigma, dt)

        # --- Statistiken aus Pfaden ---
        # Hochs und Tiefs (ueber alle Schritte inkl. Startpreis)
        highs = np.max(paths, axis=0)   # (n_simulations,)
        lows = np.min(paths, axis=0)    # (n_simulations,)
        closes = paths[-1, :]           # (n_simulations,)
        opens = paths[0, :]             # Startpreis fuer alle gleich

        # Range in %
        ranges = (highs - lows) / price * 100.0

        # Kerzenform-Statistiken
        range_abs = highs - lows
        # Vermeide Division durch Null
        safe_range = np.where(range_abs > 0, range_abs, 1e-10 * price)

        body_sizes = np.abs(closes - opens) / safe_range
        upper_wicks = (highs - np.maximum(closes, opens)) / safe_range
        lower_wicks = (np.minimum(closes, opens) - lows) / safe_range

        # Clampe auf [0, 1]
        body_sizes = np.clip(body_sizes, 0.0, 1.0)
        upper_wicks = np.clip(upper_wicks, 0.0, 1.0)
        lower_wicks = np.clip(lower_wicks, 0.0, 1.0)

        logger.info(
            f"Monte-Carlo abgeschlossen | "
            f"E[Range]={np.mean(ranges):.2f}% | "
            f"E[High]={np.mean((highs - price) / price * 100):.2f}% | "
            f"E[Low]={np.mean((lows - price) / price * 100):.2f}%"
        )

        return SimulationResult(
            highs=highs,
            lows=lows,
            closes=closes,
            ranges=ranges,
            body_sizes=body_sizes,
            upper_wicks=upper_wicks,
            lower_wicks=lower_wicks,
        )

    except Exception as e:
        logger.error(f"Monte-Carlo-Simulation fehlgeschlagen: {e}", exc_info=True)
        # Fallback: minimales valides Ergebnis
        dummy_range = state.sigma * 100.0 * np.ones(n_simulations)
        return SimulationResult(
            highs=np.full(n_simulations, state.price * (1 + state.sigma)),
            lows=np.full(n_simulations, state.price * (1 - state.sigma)),
            closes=np.full(n_simulations, state.price),
            ranges=dummy_range,
            body_sizes=np.full(n_simulations, 0.5),
            upper_wicks=np.full(n_simulations, 0.25),
            lower_wicks=np.full(n_simulations, 0.25),
        )


def _apply_gravity_correction(
    paths: np.ndarray,
    state,
    mu: float,
    sigma: float,
    dt: float
) -> None:
    """
    Korrigiert Pfade durch schwache Gravitationsanziehung zu Liquiditaets-Attraktoren.

    Die Korrektur ist absichtlich sehr schwach (gravity_force / 100) um den
    stochastischen Charakter der Simulation zu erhalten.

    Args:
        paths: (n_steps + 1, n_simulations) - wird in-place modifiziert
        state: MarketState mit liquidity_attractors
        mu, sigma, dt: GBM-Parameter
    """
    try:
        n_steps = paths.shape[0] - 1
        n_sims = paths.shape[1]

        if not state.liquidity_attractors:
            return

        from wbot.physics.liquidity_gravity import gravity_force, liquidity_density

        # Berechne Gravitationskraft fuer jeden Schritt
        # Vereinfachung: Nutze aktuelle Pfad-Positionen und naechsten Attraktor
        attraktor_preise = np.array([a[0] for a in state.liquidity_attractors])
        attraktor_staerken = np.array([a[1] for a in state.liquidity_attractors])

        # Normiere Staerken
        total_strength = attraktor_staerken.sum()
        if total_strength <= 0:
            return

        # Berechne gewichteten Attraktor-Preis als "Gravitationszentrum"
        gravity_center = float(np.sum(attraktor_preise * attraktor_staerken) / total_strength)

        # Schwache Drift-Korrektur: zieht Preis in Richtung Gravitationszentrum
        # Angewendet auf jeden Zeitschritt als additiver Log-Return-Term
        gravity_strength = 0.01  # 1% der Drift-Staerke
        correction_per_step = np.zeros(n_steps)

        for step in range(n_steps):
            current_prices = paths[step, :]  # (n_sims,)

            # Richtung zum Gravitationszentrum
            direction = (gravity_center - current_prices) / current_prices

            # Schwache Korrektur (gravity_force / 100)
            correction = gravity_strength * direction * dt / n_steps

            # Appliziere Korrektur auf Folge-Schritte (kumulativ)
            paths[step + 1:, :] *= np.exp(correction)[np.newaxis, :]

    except Exception as e:
        logger.debug(f"Gravitationskorrektur fehlgeschlagen (nicht kritisch): {e}")
