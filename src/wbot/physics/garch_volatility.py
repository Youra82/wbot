# src/wbot/physics/garch_volatility.py
# Modul 4-6: GARCH(1,1) Volatilitaetsschaetzung, Realized Vol, Regime-Erkennung
import numpy as np
import logging
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# Fallback-Parameter wenn Optimierung nicht konvergiert
GARCH_FALLBACK_OMEGA = 0.0001
GARCH_FALLBACK_ALPHA = 0.10
GARCH_FALLBACK_BETA = 0.85


def _garch_log_likelihood(params: np.ndarray, returns: np.ndarray) -> float:
    """Negative Log-Likelihood fuer GARCH(1,1)."""
    omega, alpha, beta = params
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)

    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        if sigma2[t] <= 0:
            return 1e10

    log_lik = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns ** 2 / sigma2)
    return -log_lik  # Negativ weil wir minimieren


def estimate_garch(returns: np.ndarray) -> tuple:
    """
    Schaetzt GARCH(1,1) Parameter via MLE.

    Args:
        returns: Array von Log-Returns

    Returns:
        (omega, alpha, beta) - GARCH-Parameter
    """
    if len(returns) < 30:
        logger.warning("Zu wenig Daten fuer GARCH-Schaetzung, nutze Fallback-Werte.")
        return GARCH_FALLBACK_OMEGA, GARCH_FALLBACK_ALPHA, GARCH_FALLBACK_BETA

    try:
        # Initiale Schaetzwerte
        var_r = max(np.var(returns), 1e-10)
        x0 = [var_r * 0.1, 0.1, 0.8]

        # Constraints: omega > 0, alpha > 0, beta > 0, alpha + beta < 1
        constraints = [
            {'type': 'ineq', 'fun': lambda p: p[0] - 1e-8},       # omega > 0
            {'type': 'ineq', 'fun': lambda p: p[1] - 1e-8},       # alpha > 0
            {'type': 'ineq', 'fun': lambda p: p[2] - 1e-8},       # beta > 0
            {'type': 'ineq', 'fun': lambda p: 0.9999 - p[1] - p[2]},  # alpha + beta < 1
        ]
        bounds = [(1e-8, 1.0), (1e-6, 0.5), (1e-6, 0.9999)]

        result = minimize(
            _garch_log_likelihood,
            x0,
            args=(returns,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-8, 'maxiter': 500, 'disp': False}
        )

        if result.success and all(p > 0 for p in result.x) and result.x[1] + result.x[2] < 1.0:
            omega, alpha, beta = result.x
            logger.debug(f"GARCH(1,1) konvergiert: omega={omega:.6f}, alpha={alpha:.4f}, beta={beta:.4f}")
            return float(omega), float(alpha), float(beta)
        else:
            logger.warning(f"GARCH-Optimierung nicht konvergiert (success={result.success}), nutze Fallback.")
            return GARCH_FALLBACK_OMEGA, GARCH_FALLBACK_ALPHA, GARCH_FALLBACK_BETA

    except Exception as e:
        logger.warning(f"GARCH-Schaetzung fehlgeschlagen: {e}. Nutze Fallback-Werte.")
        return GARCH_FALLBACK_OMEGA, GARCH_FALLBACK_ALPHA, GARCH_FALLBACK_BETA


def forecast_volatility(returns: np.ndarray, omega: float, alpha: float, beta: float, steps: int = 1) -> float:
    """
    Prognostiziert die Volatilitaet fuer t+steps Schritte.

    Args:
        returns: Array von Log-Returns
        omega, alpha, beta: GARCH-Parameter
        steps: Anzahl Prognose-Schritte voraus

    Returns:
        sigma_t+steps (Standardabweichung, nicht Varianz)
    """
    if len(returns) < 2:
        return float(np.std(returns)) if len(returns) > 0 else 0.01

    try:
        n = len(returns)
        # Berechne sigma2 fuer alle verfuegbaren Daten
        sigma2 = np.var(returns)
        for t in range(1, n):
            sigma2 = omega + alpha * returns[t - 1] ** 2 + beta * sigma2

        # Multi-step forecast: E[sigma^2_{t+h}] = omega/(1-alpha-beta) + (alpha+beta)^h * (sigma2 - omega/(1-alpha-beta))
        if steps > 1:
            persistence = alpha + beta
            if persistence < 1.0:
                long_run_var = omega / (1.0 - persistence)
                sigma2_forecast = long_run_var + (persistence ** steps) * (sigma2 - long_run_var)
                sigma2_forecast = max(sigma2_forecast, 1e-10)
            else:
                sigma2_forecast = sigma2
        else:
            sigma2_forecast = sigma2

        return float(np.sqrt(max(sigma2_forecast, 1e-10)))

    except Exception as e:
        logger.warning(f"Volatilitaets-Prognose fehlgeschlagen: {e}")
        return float(np.std(returns)) if len(returns) > 0 else 0.01


def realized_volatility(returns: np.ndarray, window: int = 14) -> float:
    """
    Berechnet die realisierte Volatilitaet (RV) als sqrt(sum(r^2)) ueber ein Fenster.

    Args:
        returns: Array von Log-Returns
        window: Fensterlaenge

    Returns:
        RV (annualisierungsfaehig)
    """
    if len(returns) < 2:
        return 0.0

    try:
        recent = returns[-window:] if len(returns) >= window else returns
        rv = float(np.sqrt(np.sum(recent ** 2)))
        return rv
    except Exception as e:
        logger.warning(f"Realized Volatility Berechnung fehlgeschlagen: {e}")
        return 0.0


def volatility_regime(rv: float, sigma: float) -> str:
    """
    Bestimmt das Volatilitaets-Regime als VR = RV/sigma.

    Args:
        rv: Realisierte Volatilitaet
        sigma: GARCH-prognostizierte Volatilitaet

    Returns:
        "high" | "normal" | "low"
    """
    if sigma <= 0:
        return "normal"

    try:
        vr = rv / sigma
        if vr > 1.2:
            return "high"
        elif vr < 0.8:
            return "low"
        else:
            return "normal"
    except Exception:
        return "normal"


def garman_klass_vol(
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    close_prices: np.ndarray,
    window: int = 14
) -> float:
    """
    Garman-Klass Volatilitaetsschaetzung (nutzt OHLC-Daten effizient).

    GK = sqrt( (1/(2*n)) * sum((ln(H/L))^2) - (2*ln(2)-1)/n * sum((ln(C/O))^2) )

    Args:
        open_prices, high_prices, low_prices, close_prices: OHLC-Preisarrays
        window: Fensterlaenge

    Returns:
        Garman-Klass Volatilitaet (Standardabweichung pro Kerze)
    """
    if len(close_prices) < 2:
        return 0.01

    try:
        n = min(window, len(close_prices))
        o = open_prices[-n:]
        h = high_prices[-n:]
        l = low_prices[-n:]
        c = close_prices[-n:]

        # Vermeide Division durch Null
        mask = (o > 0) & (h > 0) & (l > 0) & (c > 0) & (h >= l)
        if mask.sum() < 2:
            return float(np.std(np.log(c[1:] / c[:-1]))) if len(c) > 1 else 0.01

        o, h, l, c = o[mask], h[mask], l[mask], c[mask]
        n_valid = len(o)

        hl_term = 0.5 * np.log(h / l) ** 2
        co_term = (2 * np.log(2) - 1) * np.log(c / o) ** 2

        gk_var = np.mean(hl_term) - np.mean(co_term)
        gk_var = max(gk_var, 1e-10)

        return float(np.sqrt(gk_var))

    except Exception as e:
        logger.warning(f"Garman-Klass Vol Berechnung fehlgeschlagen: {e}")
        return 0.01
