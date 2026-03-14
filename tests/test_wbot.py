# tests/test_wbot.py
# Integrations-Tests fuer wbot QGRS
import os
import sys
import pytest
import numpy as np
import pandas as pd

# Sicherstellen dass src/ im Pfad ist
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))


# ==============================================================================
# Test 1: GARCH Volatilitaet
# ==============================================================================

def test_garch_volatility():
    """Testet GARCH-Schaetzung und Volatilitaets-Prognose."""
    from wbot.physics.garch_volatility import estimate_garch, forecast_volatility

    np.random.seed(42)
    returns = np.random.randn(200) * 0.01

    omega, alpha, beta = estimate_garch(returns)

    assert omega > 0, f"omega muss positiv sein, ist {omega}"
    assert alpha > 0, f"alpha muss positiv sein, ist {alpha}"
    assert beta > 0, f"beta muss positiv sein, ist {beta}"
    assert alpha + beta < 1.0, f"alpha+beta muss < 1 sein, ist {alpha + beta}"

    sigma = forecast_volatility(returns, omega, alpha, beta)
    assert sigma > 0, f"sigma muss positiv sein, ist {sigma}"
    assert sigma < 1.0, f"sigma ist unrealistisch gross: {sigma}"


def test_garch_fallback():
    """Testet GARCH-Fallback bei zu wenig Daten."""
    from wbot.physics.garch_volatility import estimate_garch

    # Zu wenige Daten -> Fallback-Werte
    returns = np.random.randn(5) * 0.01
    omega, alpha, beta = estimate_garch(returns)

    assert omega > 0
    assert alpha > 0
    assert beta > 0


def test_garman_klass_vol():
    """Testet Garman-Klass Volatilitaet."""
    from wbot.physics.garch_volatility import garman_klass_vol

    np.random.seed(42)
    n = 50
    closes = 50000 + np.cumsum(np.random.randn(n) * 100)
    opens = closes * (1 + np.random.randn(n) * 0.001)
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.randn(n) * 0.005))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.randn(n) * 0.005))

    gk_vol = garman_klass_vol(opens, highs, lows, closes)
    assert gk_vol > 0
    assert gk_vol < 0.1  # Max 10% Volatilitaet pro Kerze


# ==============================================================================
# Test 2: Fraktaldimension
# ==============================================================================

def test_fractal_dimension():
    """Testet Hurst-Exponent und Fraktaldimension."""
    from wbot.physics.fractal_dimension import hurst_exponent, fractal_dimension, fractal_regime

    np.random.seed(42)
    prices = np.cumprod(1 + np.random.randn(200) * 0.01) * 50000

    H = hurst_exponent(prices)
    assert 0 < H < 1, f"Hurst-Exponent muss in (0,1) sein, ist {H}"

    D = fractal_dimension(prices)
    assert 1 < D < 2, f"Fraktaldimension muss in (1,2) sein, ist {D}"

    # H + (2-D) = 2 muss gelten (D = 2 - H)
    assert abs(D - (2.0 - H)) < 0.01, f"D = 2 - H verletzt: D={D}, H={H}"

    regime = fractal_regime(D)
    assert regime in ["trend", "random", "chaos"]


def test_hurst_fallback():
    """Testet Hurst-Fallback bei zu wenig Daten."""
    from wbot.physics.fractal_dimension import hurst_exponent

    prices = np.array([100.0, 101.0, 99.0])  # Zu wenig
    H = hurst_exponent(prices)
    assert H == 0.5  # Fallback


# ==============================================================================
# Test 3: Chaos-Indikatoren
# ==============================================================================

def test_lyapunov_exponent():
    """Testet den Lyapunov-Exponenten."""
    from wbot.physics.chaos_indicators import lyapunov_exponent, attractor_regime

    np.random.seed(42)
    # Random walk sollte nahe 0 liegen
    prices = np.cumprod(1 + np.random.randn(100) * 0.01) * 50000

    lambda_val = lyapunov_exponent(prices)
    # Keine exakte Erwartung, nur Bereich pruefen
    assert isinstance(lambda_val, float), "Lyapunov muss float sein"

    regime = attractor_regime(lambda_val, 1.5)
    assert regime in ["trend", "range", "chaotic", "crash_risk"]


def test_phase_space_embedding():
    """Testet die Phasenraum-Einbettung."""
    from wbot.physics.chaos_indicators import phase_space_embedding

    prices = np.linspace(100, 200, 50)
    embedded = phase_space_embedding(prices, dim=3, lag=1)

    assert embedded.shape[1] == 3, "Einbettungsdimension muss 3 sein"
    assert len(embedded) == 50 - 2, f"Laenge falsch: {len(embedded)}"


# ==============================================================================
# Test 4: Informationsfluss
# ==============================================================================

def test_entropy():
    """Testet Shannon-Entropie."""
    from wbot.physics.information_flow import entropy

    np.random.seed(42)
    returns = np.random.randn(100) * 0.01

    H = entropy(returns, bins=20)
    assert H >= 0, f"Entropie muss >= 0 sein, ist {H}"
    assert H < 10, f"Entropie ist unrealistisch gross: {H}"

    # Uniforme Verteilung hat hoehere Entropie als konzentrierte
    uniform_returns = np.linspace(-0.1, 0.1, 100)
    H_uniform = entropy(uniform_returns, bins=20)
    # Uniform hat maximale Entropie fuer gleiche Bin-Anzahl
    assert H_uniform >= 0


def test_information_flow():
    """Testet Informationsfluss-Berechnung."""
    from wbot.physics.information_flow import information_flow

    np.random.seed(42)
    returns = np.random.randn(60) * 0.01

    phi = information_flow(returns, window=20)
    assert isinstance(phi, float), "Phi muss float sein"


# ==============================================================================
# Test 5: Liquiditaets-Gravitation
# ==============================================================================

def test_liquidity_gravity():
    """Testet Liquiditaetsdichte und Gravitationskraft."""
    from wbot.physics.liquidity_gravity import (
        liquidity_density, gravity_force, find_liquidity_attractors
    )

    np.random.seed(42)
    n = 200
    prices = 50000 + np.cumsum(np.random.randn(n) * 500)
    volumes = np.abs(np.random.randn(n) * 1e6) + 1e5

    density, price_levels = liquidity_density(prices, volumes, bins=50)

    assert len(density) == 50
    assert len(price_levels) == 50
    assert abs(density.sum() - 1.0) < 1e-6, "Dichte muss normiert sein"

    F = gravity_force(float(prices[-1]), density, price_levels)
    assert isinstance(F, float), "Gravitationskraft muss float sein"

    attractors = find_liquidity_attractors(density, price_levels, n_top=5)
    assert isinstance(attractors, list)
    assert len(attractors) <= 5


# ==============================================================================
# Test 6: Marktenergie
# ==============================================================================

def test_market_energy():
    """Testet Multi-Timeframe Marktenergie."""
    from wbot.physics.market_energy import multi_timeframe_energy, energy_regime

    np.random.seed(42)
    n = 100
    closes = 50000 + np.cumsum(np.random.randn(n) * 500)
    volumes = np.abs(np.random.randn(n) * 1e6) + 1e5

    df = pd.DataFrame({
        'open': closes * 0.999,
        'high': closes * 1.002,
        'low': closes * 0.998,
        'close': closes,
        'volume': volumes
    })

    E = multi_timeframe_energy(df, windows=[5, 14, 30])
    assert E >= 0, f"Energie muss >= 0 sein, ist {E}"

    regime = energy_regime(E)
    assert regime in ["high", "normal", "low"]


# ==============================================================================
# Test 7: MarketState
# ==============================================================================

def test_compute_market_state():
    """Testet die vollstaendige MarketState-Berechnung."""
    from wbot.model.market_state import compute_market_state

    np.random.seed(42)
    n = 150
    closes = 50000 + np.cumsum(np.random.randn(n) * 200)
    volumes = np.abs(np.random.randn(n) * 1e6) + 1e5

    df = pd.DataFrame({
        'open': closes * (1 + np.random.randn(n) * 0.001),
        'high': closes * (1 + np.abs(np.random.randn(n) * 0.005)),
        'low': closes * (1 - np.abs(np.random.randn(n) * 0.005)),
        'close': closes,
        'volume': volumes
    })

    state = compute_market_state(df)

    assert state.price > 0
    assert state.sigma > 0
    assert 1 < state.fractal_D < 2
    assert 0 < state.hurst_H < 1
    assert isinstance(state.lyapunov, float)
    assert isinstance(state.info_flow, float)
    assert isinstance(state.liquidity_force, float)
    assert isinstance(state.liquidity_attractors, list)
    assert state.energy >= 0
    assert state.vol_regime in ["high", "normal", "low"]
    assert state.attractor_regime in ["trend", "range", "chaotic", "crash_risk"]
    assert state.fractal_regime in ["trend", "random", "chaos"]


# ==============================================================================
# Test 8: Monte-Carlo Simulation
# ==============================================================================

def test_monte_carlo_basic():
    """Testet grundlegende Monte-Carlo-Eigenschaften."""
    from wbot.model.market_state import MarketState
    from wbot.model.monte_carlo import run_monte_carlo

    np.random.seed(42)
    state = MarketState(
        price=50000.0,
        returns=np.random.randn(100) * 0.01,
        sigma=0.02,
        fractal_D=1.5,
        hurst_H=0.5,
        lyapunov=0.01,
        info_flow=0.001,
        liquidity_force=0.0,
        liquidity_attractors=[],
        energy=0.0001,
        vol_regime="normal",
        attractor_regime="range",
        fractal_regime="random"
    )

    result = run_monte_carlo(state, n_simulations=100, n_steps=50)

    assert len(result.highs) == 100, "Anzahl Highs stimmt nicht"
    assert len(result.lows) == 100, "Anzahl Lows stimmt nicht"
    assert len(result.closes) == 100, "Anzahl Closes stimmt nicht"
    assert len(result.ranges) == 100, "Anzahl Ranges stimmt nicht"
    assert all(result.highs >= result.lows), "High muss >= Low sein"
    assert all(result.ranges >= 0), "Range muss >= 0 sein"
    assert all(result.body_sizes >= 0) and all(result.body_sizes <= 1), "Body-Size muss in [0,1] sein"


def test_monte_carlo_with_attractors():
    """Testet Monte-Carlo mit Liquiditaets-Attraktoren."""
    from wbot.model.market_state import MarketState
    from wbot.model.monte_carlo import run_monte_carlo

    np.random.seed(42)
    state = MarketState(
        price=50000.0,
        returns=np.random.randn(100) * 0.01,
        sigma=0.015,
        fractal_D=1.4,
        hurst_H=0.6,
        lyapunov=-0.01,
        info_flow=0.002,
        liquidity_force=0.001,
        liquidity_attractors=[(49000.0, 150.0), (51500.0, 120.0), (47000.0, 80.0)],
        energy=0.0002,
        vol_regime="normal",
        attractor_regime="trend",
        fractal_regime="trend"
    )

    result = run_monte_carlo(state, n_simulations=200, n_steps=50)

    assert len(result.highs) == 200
    assert all(result.highs >= result.lows)


# ==============================================================================
# Test 9: Range Forecast
# ==============================================================================

def test_range_forecast():
    """Testet Range-Prognose aus Monte-Carlo-Ergebnissen."""
    from wbot.model.market_state import MarketState
    from wbot.model.monte_carlo import run_monte_carlo
    from wbot.forecast.range_forecast import compute_range_forecast

    np.random.seed(42)
    state = MarketState(
        price=50000.0,
        returns=np.random.randn(100) * 0.01,
        sigma=0.02,
        fractal_D=1.5,
        hurst_H=0.5,
        lyapunov=0.01,
        info_flow=0.001,
        liquidity_force=0.0,
        liquidity_attractors=[],
        energy=0.0001,
        vol_regime="normal",
        attractor_regime="range",
        fractal_regime="random"
    )

    sim = run_monte_carlo(state, n_simulations=500, n_steps=50)
    forecast = compute_range_forecast(sim, 50000.0)

    assert 0 < forecast.expected_range_pct < 20, f"Range unrealistisch: {forecast.expected_range_pct}"
    assert 0 <= forecast.prob_bullish <= 1
    assert 0 <= forecast.prob_bearish <= 1
    assert abs(forecast.prob_bullish + forecast.prob_bearish - 1.0) < 0.1  # Grob addiert
    assert 0 <= forecast.prob_doji <= 1
    assert 0 <= forecast.prob_pinbar <= 1
    assert forecast.q50_range_pct <= forecast.q75_range_pct <= forecast.q90_range_pct
    assert forecast.expected_high_pct >= 0
    assert forecast.expected_low_pct >= 0


def test_range_forecast_table():
    """Testet die Forecast-Tabellen-Formatierung."""
    from wbot.forecast.range_forecast import RangeForecast, format_forecast_table

    forecast = RangeForecast(
        expected_range_pct=2.0,
        q50_range_pct=1.8,
        q75_range_pct=2.5,
        q90_range_pct=3.5,
        q95_range_pct=4.5,
        prob_high_1pct=0.5,
        prob_high_2pct=0.3,
        prob_high_3pct=0.15,
        prob_low_1pct=0.5,
        prob_low_2pct=0.3,
        prob_low_3pct=0.15,
        prob_bullish=0.52,
        prob_bearish=0.48,
        prob_doji=0.12,
        prob_pinbar=0.08,
        expected_high_pct=1.0,
        expected_low_pct=1.0,
    )

    table = format_forecast_table(forecast)
    assert isinstance(table, str)
    assert "Range" in table
    assert "Bullish" in table or "bullish" in table or "BULLISH" in table


# ==============================================================================
# Test 10: Kerzenform-Klassifikation
# ==============================================================================

def test_candle_shape_classification():
    """Testet Kerzenform-Klassifikation."""
    from wbot.forecast.candle_shape import classify_candle_shape, candle_shape_distribution, most_likely_shape
    from wbot.model.monte_carlo import SimulationResult

    # Doji: kleiner Koerper
    shape = classify_candle_shape(body=0.1, upper_wick=0.45, lower_wick=0.45)
    assert shape == "doji"

    # Bullisher Pinbar: grosser unterer Docht
    shape = classify_candle_shape(body=0.2, upper_wick=0.1, lower_wick=0.7)
    assert shape == "bullish_pinbar"

    # Bearisher Pinbar: grosser oberer Docht
    shape = classify_candle_shape(body=0.2, upper_wick=0.7, lower_wick=0.1)
    assert shape == "bearish_pinbar"

    # Bullish Trend: grosser Koerper, kleiner oberer Docht
    shape = classify_candle_shape(body=0.7, upper_wick=0.1, lower_wick=0.2)
    assert shape in ["bullish_trend", "bearish_trend"]

    # Test mit SimulationResult
    np.random.seed(42)
    n = 100
    sim = SimulationResult(
        highs=np.full(n, 51000.0),
        lows=np.full(n, 49000.0),
        closes=np.linspace(49500.0, 50500.0, n),
        ranges=np.full(n, 4.0),
        body_sizes=np.random.uniform(0.1, 0.9, n),
        upper_wicks=np.random.uniform(0.05, 0.45, n),
        lower_wicks=np.random.uniform(0.05, 0.45, n),
    )

    dist = candle_shape_distribution(sim)
    assert isinstance(dist, dict)
    assert abs(sum(dist.values()) - 1.0) < 1e-6, "Verteilung muss normiert sein"

    best_shape, best_prob = most_likely_shape(dist)
    assert isinstance(best_shape, str)
    assert 0 <= best_prob <= 1


# ==============================================================================
# Test 11: Phasenraum
# ==============================================================================

def test_phase_space():
    """Testet Phasenraum-Berechnung und Regime-Klassifikation."""
    from wbot.forecast.phase_space import (
        compute_phase_space_vector,
        classify_phase_space_regime,
        get_phase_space_regime_from_prices
    )

    np.random.seed(42)
    prices = 50000 + np.cumsum(np.random.randn(30) * 200)

    P, v, a, rho = compute_phase_space_vector(prices)
    assert P > 0
    assert isinstance(v, float)
    assert isinstance(a, float)
    assert 0 <= rho <= 1, f"rho muss in [0,1] sein, ist {rho}"

    regime = classify_phase_space_regime(v, a, rho)
    assert regime in ["uptrend_attractor", "downtrend_attractor", "range_zone", "crash_spike_zone"]

    regime_from_prices = get_phase_space_regime_from_prices(prices)
    assert regime_from_prices in ["uptrend_attractor", "downtrend_attractor", "range_zone", "crash_spike_zone"]


# ==============================================================================
# Test 12: Signal Logic
# ==============================================================================

def test_signal_logic():
    """Testet Signal-Generierung (kein Exchange benoetigt)."""
    from wbot.model.market_state import MarketState
    from wbot.model.monte_carlo import run_monte_carlo
    from wbot.forecast.range_forecast import compute_range_forecast
    from wbot.strategy.signal_logic import generate_signal

    np.random.seed(42)
    state = MarketState(
        price=50000.0,
        returns=np.random.randn(100) * 0.01,
        sigma=0.02,
        fractal_D=1.5,
        hurst_H=0.5,
        lyapunov=0.01,
        info_flow=0.001,
        liquidity_force=0.0,
        liquidity_attractors=[(49000, 100), (51000, 80)],
        energy=0.0001,
        vol_regime="normal",
        attractor_regime="range",
        fractal_regime="random"
    )

    sim = run_monte_carlo(state, n_simulations=200, n_steps=50)
    forecast = compute_range_forecast(sim, 50000.0)

    config = {
        "breakout_threshold_pct": 1.5,
        "breakout_prob_threshold": 0.60,
        "range_threshold_pct": 1.0,
        "range_prob_threshold": 0.65,
        "risk_per_trade_pct": 2.0,
        "leverage": 10
    }

    signal = generate_signal(forecast, state, 50000.0, config)

    assert signal.action in [
        "long_breakout", "short_breakout", "long_range", "short_range", "wait"
    ], f"Ungueltiges Signal: {signal.action}"
    assert isinstance(signal.entry_price, float)
    assert isinstance(signal.stop_loss, float)
    assert isinstance(signal.take_profit, float)
    assert 0 <= signal.confidence <= 1
    assert isinstance(signal.reason, str)


def test_signal_wait_conditions():
    """Testet Warte-Bedingungen im Signal-Generator."""
    from wbot.model.market_state import MarketState
    from wbot.model.monte_carlo import SimulationResult
    from wbot.forecast.range_forecast import RangeForecast
    from wbot.strategy.signal_logic import generate_signal

    # Chaotisches Regime mit hohem Lyapunov -> WAIT
    state_chaotic = MarketState(
        price=50000.0,
        returns=np.random.randn(100) * 0.01,
        sigma=0.05,
        fractal_D=1.8,
        hurst_H=0.2,
        lyapunov=0.8,           # Hoch!
        info_flow=0.01,
        liquidity_force=0.0,
        liquidity_attractors=[],
        energy=0.001,
        vol_regime="high",
        attractor_regime="chaotic",  # Chaotisch!
        fractal_regime="chaos"
    )

    forecast = RangeForecast(
        expected_range_pct=5.0, q50_range_pct=4.5, q75_range_pct=6.0,
        q90_range_pct=8.0, q95_range_pct=10.0,
        prob_high_1pct=0.8, prob_high_2pct=0.6, prob_high_3pct=0.4,
        prob_low_1pct=0.7, prob_low_2pct=0.5, prob_low_3pct=0.3,
        prob_bullish=0.5, prob_bearish=0.5, prob_doji=0.1, prob_pinbar=0.1,
        expected_high_pct=2.5, expected_low_pct=2.0,
    )

    config = {
        "breakout_threshold_pct": 1.5, "breakout_prob_threshold": 0.60,
        "range_threshold_pct": 1.0, "range_prob_threshold": 0.65,
        "risk_per_trade_pct": 2.0, "leverage": 10
    }

    signal = generate_signal(forecast, state_chaotic, 50000.0, config)
    assert signal.action == "wait", f"Erwarte WAIT bei chaotischem Regime, got: {signal.action}"


# ==============================================================================
# Test 13: GBM-Preisprozess
# ==============================================================================

def test_price_process():
    """Testet GBM Drift/Sigma-Berechnung und Pfad-Simulation."""
    from wbot.model.market_state import MarketState
    from wbot.model.price_process import compute_drift, compute_sigma, simulate_gbm_path

    np.random.seed(42)
    state = MarketState(
        price=50000.0,
        returns=np.random.randn(100) * 0.01,
        sigma=0.02,
        fractal_D=1.5,
        hurst_H=0.5,
        lyapunov=0.01,
        info_flow=0.001,
        liquidity_force=0.001,
        liquidity_attractors=[],
        energy=0.0001,
        vol_regime="normal",
        attractor_regime="range",
        fractal_regime="random"
    )

    mu = compute_drift(state)
    assert -0.05 <= mu <= 0.05, f"Drift ausserhalb vernuenftigen Bereichs: {mu}"

    sigma_eff = compute_sigma(state)
    assert sigma_eff > 0
    assert sigma_eff < 1.0, f"Sigma unrealistisch gross: {sigma_eff}"

    path = simulate_gbm_path(50000.0, mu, sigma_eff, n_steps=100, dt=1.0/100)
    assert len(path) == 101, f"Pfad-Laenge falsch: {len(path)}"
    assert path[0] == 50000.0, "Startpreis stimmt nicht"
    assert all(p > 0 for p in path), "Alle Pfad-Preise muessen positiv sein"
