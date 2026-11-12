"""Proper scoring rules and forecast evaluation metrics."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import norm


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Root mean squared error per dimension."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def dss(
    y_true: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Dawid–Sebastiani Score for multivariate Gaussian predictive distribution.

    Parameters
    ----------
    y_true : (..., d)
        Realized value(s).
    mean : (..., d)
        Predictive mean.
    cov : (..., d, d)
        Predictive covariance. Must be positive definite.

    Returns
    -------
    score : (...,)
        DSS values (lower is better).
    """
    diff = y_true - mean
    # Add small diagonal jitter for numerical stability if needed
    cov = np.asarray(cov)
    if eps:
        cov = cov + np.eye(cov.shape[-1]) * eps
    sign, logdet = np.linalg.slogdet(cov)
    if np.any(sign <= 0):
        raise ValueError("Covariance is not positive definite.")
    inv = np.linalg.inv(cov)
    mahal = np.einsum("...i,...ij,...j->...", diff, inv, diff)
    return logdet + mahal


def energy_score(
    y_true: np.ndarray,
    scenarios: np.ndarray,
    *,
    n_scenarios: int | None = None,
) -> np.ndarray:
    """Energy Score for a multivariate predictive distribution represented by scenarios.

    Parameters
    ----------
    y_true : (..., d)
        Realized value(s).
    scenarios : (..., n, d)
        Predictive scenarios.
    n_scenarios : int, optional
        Number of scenarios (last dim of ``scenarios``). Inferred if ``None``.

    Returns
    -------
    score : (...,)
        Energy Score values (lower is better).
    """
    if n_scenarios is None:
        n_scenarios = scenarios.shape[-2]
    # ||y - X||_2 averaged over scenarios
    diff_true = scenarios - y_true[..., None, :]
    term1 = np.mean(np.linalg.norm(diff_true, axis=-1), axis=-1)
    # ||X_i - X_j||_2 averaged over pairs
    pairwise = np.linalg.norm(
        scenarios[..., :, None, :] - scenarios[..., None, :, :], axis=-1
    )
    term2 = np.sum(pairwise, axis=(-2, -1)) / (n_scenarios ** 2)
    return term1 - 0.5 * term2


def prediction_interval_coverage(
    y_true: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    *,
    levels: Sequence[float] = (0.68, 0.90, 0.95),
) -> dict[str, np.ndarray]:
    """Marginal prediction interval coverage for Gaussian forecasts.

    Returns
    -------
    dict mapping ``"level"`` string to coverage arrays of shape ``(d,)``.
    """
    std = np.sqrt(np.diagonal(cov, axis1=-2, axis2=-1))
    z = norm.ppf(0.5 + np.array(levels) / 2)
    out = {}
    for level, zz in zip(levels, z):
        lower = mean - zz * std
        upper = mean + zz * std
        hit = (y_true >= lower) & (y_true <= upper)
        out[f"coverage_{level:.0%}"] = np.mean(hit, axis=0)
    return out
