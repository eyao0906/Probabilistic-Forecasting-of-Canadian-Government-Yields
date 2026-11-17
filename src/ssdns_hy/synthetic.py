"""Synthetic data generation from the declared joint state-space models."""

from __future__ import annotations

import numpy as np

from .loadings import dns_loadings


def simulate_yields(
    n_observations: int,
    maturities_years: np.ndarray,
    decay_per_year: float,
    factor_mean: np.ndarray,
    factor_persistence: np.ndarray,
    measurement_variance: np.ndarray,
    rng: np.random.Generator,
    model: str,
    log_variance_mean: np.ndarray,
    volatility_persistence: np.ndarray | None = None,
    volatility_variance: np.ndarray | None = None,
    initial_factor: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Simulate yields, latent factors, and optional factor log-volatility."""
    if model not in {"SS_M3", "SS_M4"}:
        raise ValueError("model must be 'SS_M3' or 'SS_M4'")
    if n_observations < 2:
        raise ValueError("n_observations must be at least two")
    tau = np.asarray(maturities_years, dtype=float)
    mean = np.asarray(factor_mean, dtype=float)
    persistence = np.asarray(factor_persistence, dtype=float)
    r = np.asarray(measurement_variance, dtype=float)
    m = np.asarray(log_variance_mean, dtype=float)
    if tau.ndim != 1 or mean.shape != (3,) or persistence.shape != (3,) or m.shape != (3,):
        raise ValueError("maturities must be a vector and factor parameters must have length 3")
    if r.shape != tau.shape or np.any(r <= 0):
        raise ValueError("measurement_variance must be positive with one entry per maturity")
    if not np.isfinite(tau).all() or not np.isfinite(mean).all() or not np.isfinite(r).all():
        raise ValueError("synthetic model parameters must be finite")
    if np.any(np.abs(persistence) >= 1):
        raise ValueError("factor persistence must lie strictly inside (-1, 1)")
    if model == "SS_M4":
        if volatility_persistence is None or volatility_variance is None:
            raise ValueError("SS_M4 requires volatility persistence and variance")
        phi = np.asarray(volatility_persistence, dtype=float)
        v = np.asarray(volatility_variance, dtype=float)
        if phi.shape != (3,) or v.shape != (3,):
            raise ValueError("SV parameter vectors must have length 3")
        if np.any(np.abs(phi) >= 1) or np.any(v <= 0):
            raise ValueError("SV persistence must be stationary and variance positive")
    else:
        phi = np.zeros(3)
        v = np.ones(3)

    loadings = dns_loadings(tau, decay_per_year)
    factors = np.empty((n_observations, 3))
    log_volatility = np.empty_like(factors)
    state = mean.copy() if initial_factor is None else np.asarray(initial_factor, dtype=float)
    if state.shape != (3,) or not np.isfinite(state).all():
        raise ValueError("initial_factor must be a finite vector of length 3")
    if model == "SS_M4":
        omega = m - v / 2.0
        h = omega + np.sqrt(v) * rng.standard_normal(3)
    for t in range(n_observations):
        if model == "SS_M3":
            variance_t = np.exp(m)
            log_volatility[t] = m
        else:
            if t:
                h = omega + phi * (h - omega) + np.sqrt(v * (1.0 - phi**2)) * (
                    rng.standard_normal(3)
                )
            variance_t = np.exp(h)
            log_volatility[t] = h
        state = mean + persistence * (state - mean) + np.sqrt(variance_t) * (
            rng.standard_normal(3)
        )
        factors[t] = state
    yields = factors @ loadings.T + np.sqrt(r) * rng.standard_normal(
        (n_observations, len(tau))
    )
    return {
        "yields": yields,
        "factors": factors,
        "log_volatility": log_volatility,
    }
