"""Conditional factor paths and predictive simulation."""

import numpy as np


def sample_factors(y, h, mu, rho, q, r, initial_mean, initial_covariance, rng):
    """Draw states conditional on parameters and observations by backward sampling."""
    n = len(y)
    pm, pc = np.empty((n, 3)), np.empty((n, 3, 3))
    fm, fc = np.empty_like(pm), np.empty_like(pc)
    mean, cov = initial_mean.copy(), initial_covariance.copy()
    a = np.diag(rho)
    for t in range(n):
        mean = mu + rho * (mean - mu)
        cov = a @ cov @ a + np.diag(q[t])
        pm[t], pc[t] = mean, cov
        s = h @ cov @ h.T + np.diag(r)
        gain = np.linalg.solve(s, h @ cov).T
        mean = mean + gain @ (y[t] - h @ mean)
        residual = np.eye(3) - gain @ h
        cov = residual @ cov @ residual.T + (gain * r) @ gain.T
        cov = (cov + cov.T) / 2
        fm[t], fc[t] = mean, cov
    states = np.empty_like(fm)
    states[-1] = rng.multivariate_normal(fm[-1], fc[-1], check_valid="raise")
    for t in range(n - 2, -1, -1):
        gain = np.linalg.solve(pc[t + 1], a @ fc[t]).T
        mean = fm[t] + gain @ (states[t + 1] - pm[t + 1])
        cov = fc[t] - gain @ pc[t + 1] @ gain.T
        states[t] = rng.multivariate_normal(mean, (cov + cov.T) / 2, check_valid="raise")
    return states


def volatility_path(mean, persistence, variance, length, rng):
    """Generate a stationary Gaussian log-variance AR(1) path."""
    if length < 1 or not (-1 < persistence < 1) or variance <= 0:
        raise ValueError("Invalid stationary log-volatility parameters")
    path = np.empty(length)
    path[0] = mean + np.sqrt(variance) * rng.standard_normal()
    scale = np.sqrt(variance * (1 - persistence**2))
    for t in range(1, length):
        path[t] = mean + persistence * (path[t - 1] - mean) + scale * rng.standard_normal()
    return path


def simulate_forecast(state, mu, rho, q, r, h, horizon, rng):
    """Simulate future factors and observed yields under constant volatility."""
    if horizon < 1 or np.any(q <= 0) or np.any(r <= 0):
        raise ValueError("Horizon and variances must be positive")
    future = np.empty((horizon, len(r)))
    state = state.copy()
    for t in range(horizon):
        state = mu + rho * (state - mu) + np.sqrt(q) * rng.standard_normal(3)
        future[t] = h @ state + np.sqrt(r) * rng.standard_normal(len(r))
    return future
