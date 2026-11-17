"""Posterior predictive eight-yield scenario generation."""

from __future__ import annotations

import numpy as np

from .kalman import _check_spd
from .loadings import dns_loadings
from .model import stochastic_innovation_covariance
from .sampler import ChainDraws


def posterior_predictive_scenarios(
    posterior: ChainDraws,
    maturities_years: np.ndarray,
    horizon_observations: int,
    rng: np.random.Generator,
    n_scenarios: int,
) -> np.ndarray:
    """Simulate full-curve target-date yields from retained joint posterior draws."""
    tau = np.asarray(maturities_years, dtype=float)
    if horizon_observations < 1 or n_scenarios < 2:
        raise ValueError("horizon_observations must be positive and n_scenarios >= 2")
    if tau.ndim != 1 or not np.isfinite(tau).all():
        raise ValueError("maturities_years must be a finite vector")
    n_draws = len(posterior.decay_per_year)
    if n_draws == 0:
        raise ValueError("posterior contains no retained draws")
    scenarios = np.empty((n_scenarios, len(tau)))
    draw_indices = rng.integers(n_draws, size=n_scenarios)
    for i, index in enumerate(draw_indices):
        mean = posterior.factor_mean[index]
        persistence = posterior.persistence[index]
        state = posterior.terminal_factor[index].copy()
        if posterior.model == "SS_M3":
            variance = np.exp(posterior.log_variance[index])
            for _ in range(horizon_observations):
                state = mean + persistence * (state - mean) + np.sqrt(variance) * (
                    rng.standard_normal(3)
                )
        elif posterior.model == "SS_M4":
            log_variance = posterior.terminal_log_variance[index].copy()
            volatility_mean = posterior.log_variance[index]
            phi = posterior.volatility_persistence[index]
            v = posterior.volatility_variance[index]
            for _ in range(horizon_observations):
                log_variance = (
                    volatility_mean
                    - v / 2.0
                    + phi * (log_variance - volatility_mean + v / 2.0)
                    + np.sqrt(v * (1.0 - phi**2)) * rng.standard_normal(3)
                )
                variance = np.diag(stochastic_innovation_covariance(log_variance))
                state = mean + persistence * (state - mean) + np.sqrt(
                    variance
                ) * rng.standard_normal(3)
        else:
            raise ValueError(f"unknown posterior model: {posterior.model}")
        loadings = dns_loadings(tau, posterior.decay_per_year[index])
        measurement_variance = posterior.measurement_variance[index]
        scenarios[i] = (
            loadings @ state
            + np.sqrt(measurement_variance) * rng.standard_normal(len(tau))
        )
    return scenarios


def predictive_moments(scenarios: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Monte Carlo mean and covariance, requiring an SPD estimate."""
    values = np.asarray(scenarios, dtype=float)
    if values.ndim != 2 or values.shape[0] < values.shape[1] + 1:
        raise ValueError("scenarios must have at least dimension + 1 rows")
    if not np.isfinite(values).all():
        raise ValueError("scenarios must be finite")
    mean = values.mean(axis=0)
    covariance = np.cov(values, rowvar=False, ddof=1)
    _check_spd("predictive covariance", covariance)
    return mean, covariance
