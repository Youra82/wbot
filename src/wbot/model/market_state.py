# src/wbot/model/market_state.py
# Modul 17: Zustandsvektor X_t - berechnet alle Physics-Features aus OHLCV-Daten
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MarketState:
    """Vollstaendiger Markt-Zustandsvektor X_t."""
    price: float
    returns: np.ndarray                  # letzte N log-returns
    sigma: float                         # GARCH-Volatilitaet
    fractal_D: float                     # Fraktaldimension
    hurst_H: float                       # Hurst-Exponent
    lyapunov: float                      # Lyapunov-Exponent
    info_flow: float                     # Phi_t (Informationsfluss)
    liquidity_force: float               # F_L am aktuellen Preis
    liquidity_attractors: list           # [(price, strength), ...]
    energy: float                        # Marktenergie
    vol_regime: str                      # "high" | "normal" | "low"
    attractor_regime: str                # "trend" | "range" | "chaotic" | "crash_risk"
    fractal_regime: str                  # "trend" | "random" | "chaos"


def compute_market_state(ohlcv_df: pd.DataFrame, config: dict = None) -> MarketState:
    """
    Berechnet den vollstaendigen Markt-Zustandsvektor aus den letzten 200 Kerzen.

    Args:
        ohlcv_df: DataFrame mit Spalten [timestamp, open, high, low, close, volume]
                  (mindestens 100 Kerzen empfohlen, 200 optimal)
        config: Optionale Konfiguration (physics-Sektion aus settings.json)

    Returns:
        MarketState mit allen berechneten Features
    """
    if config is None:
        config = {}

    physics_cfg = config.get('physics', {})
    garch_window = physics_cfg.get('garch_window', 100)
    fractal_window = physics_cfg.get('fractal_window', 100)
    chaos_window = physics_cfg.get('chaos_window', 50)
    info_window = physics_cfg.get('info_window', 30)
    liq_bins = physics_cfg.get('liquidity_bins', 100)
    energy_windows = physics_cfg.get('energy_windows', [5, 14, 30])

    # Verwende letzte 200 Kerzen
    df = ohlcv_df.tail(200).copy()

    if len(df) < 10:
        logger.error("Zu wenig OHLCV-Daten fuer MarketState-Berechnung (min. 10).")
        raise ValueError("Mindestens 10 OHLCV-Kerzen benoetigt.")

    closes = df['close'].values.astype(float)
    opens = df['open'].values.astype(float)
    highs = df['high'].values.astype(float)
    lows = df['low'].values.astype(float)
    volumes = df['volume'].values.astype(float)

    current_price = float(closes[-1])

    # Log-Returns berechnen
    log_returns = np.diff(np.log(closes + 1e-10))

    logger.info(f"Berechne MarketState fuer Preis={current_price:.2f} | {len(df)} Kerzen")

    # --- Modul 1: GARCH-Volatilitaet ---
    try:
        from wbot.physics.garch_volatility import (
            estimate_garch, forecast_volatility, realized_volatility,
            volatility_regime, garman_klass_vol
        )
        garch_returns = log_returns[-garch_window:] if len(log_returns) >= garch_window else log_returns
        omega, alpha, beta = estimate_garch(garch_returns)
        sigma = forecast_volatility(garch_returns, omega, alpha, beta, steps=1)
        rv = realized_volatility(log_returns, window=14)
        vol_reg = volatility_regime(rv, sigma)
        gk_vol = garman_klass_vol(opens, highs, lows, closes, window=14)
        sigma = max(sigma, gk_vol)  # Nutze Maximum fuer konservativere Schaetzung
        logger.info(f"  GARCH: sigma={sigma:.6f}, rv={rv:.6f}, vol_regime={vol_reg}")
    except Exception as e:
        logger.warning(f"GARCH-Modul Fehler: {e}. Nutze Fallback.")
        sigma = float(np.std(log_returns[-14:])) if len(log_returns) >= 14 else 0.01
        vol_reg = "normal"

    # --- Modul 2: Fraktaldimension & Hurst ---
    try:
        from wbot.physics.fractal_dimension import hurst_exponent, fractal_dimension, fractal_regime
        fractal_prices = closes[-fractal_window:] if len(closes) >= fractal_window else closes
        H = hurst_exponent(fractal_prices)
        D = fractal_dimension(fractal_prices)
        frac_reg = fractal_regime(D)
        logger.info(f"  Fractal: H={H:.4f}, D={D:.4f}, regime={frac_reg}")
    except Exception as e:
        logger.warning(f"Fraktal-Modul Fehler: {e}. Nutze Fallback.")
        H, D = 0.5, 1.5
        frac_reg = "random"

    # --- Modul 3: Chaos-Indikatoren ---
    try:
        from wbot.physics.chaos_indicators import lyapunov_exponent, attractor_regime
        chaos_prices = closes[-chaos_window:] if len(closes) >= chaos_window else closes
        lambda_val = lyapunov_exponent(chaos_prices, embedding_dim=3, lag=1)
        attr_reg = attractor_regime(lambda_val, D)
        logger.info(f"  Chaos: lambda={lambda_val:.4f}, attractor_regime={attr_reg}")
    except Exception as e:
        logger.warning(f"Chaos-Modul Fehler: {e}. Nutze Fallback.")
        lambda_val = 0.0
        attr_reg = "range"

    # --- Modul 4: Informationsfluss ---
    try:
        from wbot.physics.information_flow import information_flow
        phi = information_flow(log_returns, window=info_window)
        logger.info(f"  InfoFlow: phi={phi:.6f}")
    except Exception as e:
        logger.warning(f"Informationsfluss-Modul Fehler: {e}. Nutze Fallback.")
        phi = 0.0

    # --- Modul 5: Liquiditaets-Gravitationsfeld ---
    try:
        from wbot.physics.liquidity_gravity import (
            liquidity_density, gravity_force, find_liquidity_attractors
        )
        density, price_levels = liquidity_density(closes, volumes, bins=liq_bins)
        F_L = gravity_force(current_price, density, price_levels)
        attractors = find_liquidity_attractors(density, price_levels, n_top=5)
        logger.info(f"  Liquidity: F_L={F_L:.6f}, #attractors={len(attractors)}")
    except Exception as e:
        logger.warning(f"Liquiditaets-Modul Fehler: {e}. Nutze Fallback.")
        F_L = 0.0
        attractors = []

    # --- Modul 6: Marktenergie ---
    try:
        from wbot.physics.market_energy import multi_timeframe_energy
        E_total = multi_timeframe_energy(df, windows=energy_windows)
        logger.info(f"  MarketEnergy: E_total={E_total:.8f}")
    except Exception as e:
        logger.warning(f"Marktenergie-Modul Fehler: {e}. Nutze Fallback.")
        E_total = 0.0

    state = MarketState(
        price=current_price,
        returns=log_returns,
        sigma=sigma,
        fractal_D=D,
        hurst_H=H,
        lyapunov=lambda_val,
        info_flow=phi,
        liquidity_force=F_L,
        liquidity_attractors=attractors,
        energy=E_total,
        vol_regime=vol_reg,
        attractor_regime=attr_reg,
        fractal_regime=frac_reg,
    )

    logger.info(
        f"MarketState OK | price={state.price:.2f} | sigma={state.sigma:.6f} | "
        f"H={state.hurst_H:.3f} | D={state.fractal_D:.3f} | "
        f"lambda={state.lyapunov:.4f} | vol_reg={state.vol_regime} | "
        f"attr_reg={state.attractor_regime} | frac_reg={state.fractal_regime}"
    )
    return state
