"""Exact conditional Gaussian likelihood with diagonal state and observation noise."""

import numpy as np
from numba import njit


def loadings(maturities: np.ndarray, decay: float) -> np.ndarray:
    maturities = np.asarray(maturities, dtype=float)
    if maturities.ndim != 1 or not np.isfinite(maturities).all() or np.any(maturities < 0):
        raise ValueError("Maturities must be a finite nonnegative vector")
    if not np.isfinite(decay) or decay <= 0:
        raise ValueError("Decay must be positive and finite")
    x = maturities * decay
    slope = np.ones_like(x)
    positive = x > 0
    slope[positive] = -np.expm1(-x[positive]) / x[positive]
    return np.column_stack((np.ones_like(x), slope, slope - np.exp(-x)))


@njit
def filter_likelihood(y, h, mu, rho, q, r, initial_mean, initial_covariance):
    """Integrate factor states; q[t] is the transition variance into y[t]."""
    t_count, n = y.shape
    mean = initial_mean.copy()
    covariance = initial_covariance.copy()
    ll = 0.0
    for t in range(t_count):
        mean = mu + rho * (mean - mu)
        covariance = rho[:, None] * covariance * rho[None, :] + np.diag(q[t])
        innovation = y[t] - h @ mean
        s = h @ covariance @ h.T + np.diag(r)
        chol = np.linalg.cholesky(s)
        whitened = np.linalg.solve(chol, innovation)
        gain = np.linalg.solve(chol.T, np.linalg.solve(chol, h @ covariance)).T
        mean = mean + gain @ innovation
        identity = np.eye(3) - gain @ h
        covariance = identity @ covariance @ identity.T + (gain * r) @ gain.T
        covariance = (covariance + covariance.T) * 0.5
        ll -= 0.5 * (n * np.log(2 * np.pi) + 2 * np.log(np.diag(chol)).sum()
                     + whitened @ whitened)
    return ll
