# src/wbot/model/price_process.py
# Modul 18: GBM-Preisprozess mit physik-basiertem Drift und Volatilitaet
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_drift(
    state,
    a0: float = 0.0,
    a1: float = 0.3,
    a2: float = 0.2,
    a3: float = 0.1
) -> float:
    """
    Berechnet den effektiven Drift mu aus dem Markt-Zustandsvektor.

    mu = a0 + a1*phi_norm + a2*F_L_norm + a3*E_norm

    Features werden auf [-1, 1] normiert bevor sie kombiniert werden.

    Args:
        state: MarketState Objekt
        a0: Basis-Drift (typisch 0 fuer neutrale Erwartung)
        a1: Gewicht fuer Informationsfluss
        a2: Gewicht fuer Liquiditaetskraft
        a3: Gewicht fuer Marktenergie

    Returns:
        mu (Drift pro Zeitschritt, typisch im Bereich [-0.01, 0.01])
    """
    try:
        # Normierung Phi: Informationsfluss typisch in [-0.5, 0.5]
        phi_norm = float(np.tanh(state.info_flow * 10.0))

        # Normierung F_L: Liquiditaetskraft (normiert mit tanh)
        fl_norm = float(np.tanh(state.liquidity_force * 100.0))

        # Normierung E: Marktenergie - log-transformiert und normiert
        if state.energy > 0:
            e_log = np.log1p(state.energy * 1e6)
            e_norm = float(np.tanh(e_log / 5.0))
        else:
            e_norm = 0.0

        mu = a0 + a1 * phi_norm + a2 * fl_norm + a3 * e_norm

        # Clampe Drift auf vernuenftige Werte (max +-2% pro Tag)
        mu = float(np.clip(mu, -0.02, 0.02))

        logger.debug(
            f"Drift: mu={mu:.6f} | phi_norm={phi_norm:.4f} | "
            f"fl_norm={fl_norm:.4f} | e_norm={e_norm:.4f}"
        )
        return mu

    except Exception as e:
        logger.warning(f"Drift-Berechnung fehlgeschlagen: {e}. Nutze 0.")
        return 0.0


def compute_sigma(
    state,
    b1: float = 0.3,
    b2: float = 0.2
) -> float:
    """
    Berechnet die effektive Volatilitaet sigma_eff aus dem Zustandsvektor.

    sigma_eff = sigma * (1 + b1*(D-1.5) + b2*max(0, lambda))

    - Hoehere Fraktaldimension (D > 1.5) erhoehlt sigma
    - Positiver Lyapunov-Exponent erhoehlt sigma (chaotisches Regime)

    Args:
        state: MarketState Objekt
        b1: Gewicht fuer Fraktaldimensions-Modifikator
        b2: Gewicht fuer Lyapunov-Modifikator

    Returns:
        sigma_eff (effektive Volatilitaet)
    """
    try:
        base_sigma = max(state.sigma, 1e-6)

        # Fraktaldimensions-Modifikator: D=1.5 => neutral, D>1.5 => mehr Vol
        D_modifier = b1 * (state.fractal_D - 1.5)

        # Chaos-Modifikator: nur wenn positiver Lyapunov
        chaos_modifier = b2 * max(0.0, state.lyapunov)

        sigma_eff = base_sigma * (1.0 + D_modifier + chaos_modifier)

        # Minimum: 0.1% pro Tag; Maximum: 20% pro Tag
        sigma_eff = float(np.clip(sigma_eff, 0.001, 0.20))

        logger.debug(
            f"Sigma: sigma_eff={sigma_eff:.6f} | base={base_sigma:.6f} | "
            f"D_mod={D_modifier:.4f} | chaos_mod={chaos_modifier:.4f}"
        )
        return sigma_eff

    except Exception as e:
        logger.warning(f"Sigma-Berechnung fehlgeschlagen: {e}. Nutze base sigma.")
        return max(float(state.sigma), 0.01) if state.sigma > 0 else 0.01


def simulate_gbm_path(
    price: float,
    mu: float,
    sigma: float,
    n_steps: int = 288,
    dt: float = 1.0 / 288
) -> np.ndarray:
    """
    Simuliert einen einzelnen GBM-Pfad (Geometrische Brownsche Bewegung).

    P_t+dt = P_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

    Args:
        price: Startpreis
        mu: Drift (pro Zeiteinheit)
        sigma: Volatilitaet (pro Zeiteinheit, annualisiert)
        n_steps: Anzahl Simulationsschritte (288 = 5-Min-Intervalle pro Tag)
        dt: Zeitschrittgroesse (1/288 = ein 5-Min-Schritt als Bruchteil eines Tages)

    Returns:
        numpy Array der Laenge n_steps + 1 (inklusive Startpreis)
    """
    try:
        Z = np.random.standard_normal(n_steps)
        drift_term = (mu - 0.5 * sigma ** 2) * dt
        diffusion_term = sigma * np.sqrt(dt) * Z

        increments = np.exp(drift_term + diffusion_term)

        path = np.empty(n_steps + 1)
        path[0] = price
        path[1:] = price * np.cumprod(increments)

        return path

    except Exception as e:
        logger.warning(f"GBM-Pfad Simulation fehlgeschlagen: {e}")
        return np.full(n_steps + 1, price)
