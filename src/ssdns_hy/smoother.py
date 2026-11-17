"""Rauch-Tung-Striebel smoothing and full-path Gaussian simulation."""

from __future__ import annotations

import numpy as np

from .kalman import FilterResult


def _symmetric(value: np.ndarray) -> np.ndarray:
    return (value + value.T) / 2


def _validate_transition(result: FilterResult, transition: np.ndarray) -> np.ndarray:
    a = np.asarray(transition, dtype=float)
    n_state = result.filtered_mean.shape[1]
    if a.shape != (n_state, n_state) or not np.isfinite(a).all():
        raise ValueError("transition must be a finite square state-dimension matrix")
    if result.filtered_mean.shape[0] == 0:
        raise ValueError("cannot smooth an empty state sequence")
    return a


def _smoothing_gains(
    result: FilterResult, transition: np.ndarray
) -> np.ndarray:
    a = _validate_transition(result, transition)
    n_time, n_state = result.filtered_mean.shape
    gains = np.empty((max(n_time - 1, 0), n_state, n_state))
    for t in range(n_time - 1):
        rhs = a @ result.filtered_covariance[t]
        gains[t] = np.linalg.solve(result.predicted_covariance[t + 1], rhs).T
    return gains


def rts_smoother(
    result: FilterResult, transition: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact conditional means and marginal covariances for all states."""
    a = _validate_transition(result, transition)
    means = result.filtered_mean.copy()
    covariances = result.filtered_covariance.copy()
    gains = _smoothing_gains(result, a)
    for t in range(len(means) - 2, -1, -1):
        gain = gains[t]
        means[t] += gain @ (means[t + 1] - result.predicted_mean[t + 1])
        covariances[t] += gain @ (
            covariances[t + 1] - result.predicted_covariance[t + 1]
        ) @ gain.T
        covariances[t] = _symmetric(covariances[t])
    return means, covariances


def sample_state_path(
    result: FilterResult, transition: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Draw the complete state path conditional on all observations."""
    a = _validate_transition(result, transition)
    gains = _smoothing_gains(result, a)
    n_time, n_state = result.filtered_mean.shape
    path = np.empty((n_time, n_state))
    chol_terminal = np.linalg.cholesky(result.filtered_covariance[-1])
    path[-1] = result.filtered_mean[-1] + chol_terminal @ rng.standard_normal(n_state)

    for t in range(n_time - 2, -1, -1):
        gain = gains[t]
        conditional_mean = result.filtered_mean[t] + gain @ (
            path[t + 1] - result.predicted_mean[t + 1]
        )
        conditional_covariance = _symmetric(
            result.filtered_covariance[t]
            - gain @ result.predicted_covariance[t + 1] @ gain.T
        )
        chol = np.linalg.cholesky(conditional_covariance)
        path[t] = conditional_mean + chol @ rng.standard_normal(n_state)
    return path
