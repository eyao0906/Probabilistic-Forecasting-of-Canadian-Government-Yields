"""State and covariance equations for the diagonal-factor DNS specification."""

from __future__ import annotations

import numpy as np


def factor_dynamics(
    long_run_mean: np.ndarray, persistence: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return A and c for f_t = A f_(t-1) + c + eta_t."""
    mean = np.asarray(long_run_mean, dtype=float)
    rho = np.asarray(persistence, dtype=float)
    if mean.ndim != 1 or mean.size == 0 or rho.shape != mean.shape:
        raise ValueError("long_run_mean and persistence must be nonempty vectors of equal shape")
    if not np.isfinite(mean).all() or not np.isfinite(rho).all():
        raise ValueError("factor means and persistence must be finite")
    if np.any(np.abs(rho) >= 1):
        raise ValueError("factor persistence must lie strictly inside (-1, 1)")
    transition = np.diag(rho)
    intercept = (1.0 - rho) * mean
    return transition, intercept


def factor_innovation_covariance(variances: np.ndarray) -> np.ndarray:
    """Return the diagonal constant factor-innovation covariance."""
    q = np.asarray(variances, dtype=float)
    if q.ndim != 1 or q.size == 0 or not np.isfinite(q).all() or np.any(q <= 0):
        raise ValueError("factor variances must be a nonempty positive finite vector")
    return np.diag(q)


def stochastic_innovation_covariance(log_variances: np.ndarray) -> np.ndarray:
    """Return diag(exp(h)) for one time point, rejecting overflow or underflow."""
    h = np.asarray(log_variances, dtype=float)
    if h.ndim != 1 or h.size == 0 or not np.isfinite(h).all():
        raise ValueError("log_variances must be a nonempty finite vector")
    with np.errstate(over="raise", under="raise", invalid="raise"):
        variances = np.exp(h)
    if not np.isfinite(variances).all() or np.any(variances <= 0):
        raise FloatingPointError("log variances do not produce finite positive variances")
    return np.diag(variances)
