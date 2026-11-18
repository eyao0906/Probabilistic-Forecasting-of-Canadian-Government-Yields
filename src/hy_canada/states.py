"""Forward simulation conditional on a posterior parameter and state draw."""

import numpy as np


def simulate_forecast(state, mu, rho, q, r, loadings, horizon, rng):
    """Generate yields with full factor covariance and diagonal measurement noise."""
    if horizon < 1 or np.any(r <= 0):
        raise ValueError("Horizon and measurement variances must be positive")
    factor_chol = np.linalg.cholesky(q)
    state = state.copy()
    yields = np.empty((horizon, len(r)))
    for t in range(horizon):
        state = mu + rho * (state - mu) + factor_chol @ rng.standard_normal(3)
        yields[t] = loadings @ state + np.sqrt(r) * rng.standard_normal(len(r))
    return yields
