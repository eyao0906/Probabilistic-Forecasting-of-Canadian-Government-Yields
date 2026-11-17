"""Cholesky-based Kalman likelihood for a linear Gaussian state-space model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FilterResult:
    """One-step filter quantities for states observed after each transition."""

    log_likelihood: float
    predicted_mean: np.ndarray
    predicted_covariance: np.ndarray
    filtered_mean: np.ndarray
    filtered_covariance: np.ndarray
    innovations: np.ndarray
    innovation_covariance: np.ndarray
    whitened_innovations: np.ndarray


def _finite_array(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _check_spd(name: str, value: np.ndarray) -> np.ndarray:
    covariance = _finite_array(name, value)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    np.linalg.cholesky(covariance)
    return covariance


def kalman_filter(
    observations: np.ndarray,
    loadings: np.ndarray,
    transition: np.ndarray,
    intercept: np.ndarray,
    process_covariance: np.ndarray,
    measurement_covariance: np.ndarray,
    initial_mean: np.ndarray,
    initial_covariance: np.ndarray,
) -> FilterResult:
    """Filter yields whose state at each date follows an AR transition.

    ``initial_mean`` and ``initial_covariance`` describe the state before the
    first observation. ``process_covariance`` is either one covariance matrix
    or a time-indexed ``(T, K, K)`` array; entry zero governs the transition
    into the first observed state.
    """
    y = _finite_array("observations", observations)
    h = _finite_array("loadings", loadings)
    a = _finite_array("transition", transition)
    c = _finite_array("intercept", intercept)
    m0 = _finite_array("initial_mean", initial_mean)
    p0 = _check_spd("initial_covariance", initial_covariance)
    r = _check_spd("measurement_covariance", measurement_covariance)

    if y.ndim != 2 or y.shape[0] == 0:
        raise ValueError("observations must have shape (T, number_of_maturities), T >= 1")
    n_time, n_yield = y.shape
    if h.ndim != 2 or h.shape[0] != n_yield:
        raise ValueError("loadings must have one row per observed yield")
    n_state = h.shape[1]
    if a.shape != (n_state, n_state) or c.shape != (n_state,):
        raise ValueError("transition/intercept dimensions do not match the state dimension")
    if r.shape != (n_yield, n_yield):
        raise ValueError("measurement_covariance dimensions do not match the yields")
    if m0.shape != (n_state,) or p0.shape != (n_state, n_state):
        raise ValueError("initial-state dimensions do not match the state dimension")

    q = _finite_array("process_covariance", process_covariance)
    if q.shape == (n_state, n_state):
        q = np.broadcast_to(q, (n_time, n_state, n_state))
    if q.shape != (n_time, n_state, n_state):
        raise ValueError("process_covariance must be (K, K) or (T, K, K)")
    for time_index in range(n_time):
        _check_spd(f"process_covariance[{time_index}]", q[time_index])

    predicted_mean = np.empty((n_time, n_state))
    predicted_covariance = np.empty((n_time, n_state, n_state))
    filtered_mean = np.empty_like(predicted_mean)
    filtered_covariance = np.empty_like(predicted_covariance)
    innovations = np.empty((n_time, n_yield))
    innovation_covariance = np.empty((n_time, n_yield, n_yield))
    whitened_innovations = np.empty_like(innovations)
    log_likelihood = 0.0
    mean_previous, covariance_previous = m0, p0
    identity = np.eye(n_state)

    for time_index in range(n_time):
        mean_predicted = a @ mean_previous + c
        covariance_predicted = a @ covariance_previous @ a.T + q[time_index]
        covariance_predicted = (covariance_predicted + covariance_predicted.T) / 2
        innovation = y[time_index] - h @ mean_predicted
        covariance_innovation = h @ covariance_predicted @ h.T + r
        covariance_innovation = (covariance_innovation + covariance_innovation.T) / 2
        chol = np.linalg.cholesky(covariance_innovation)
        whitened = np.linalg.solve(chol, innovation)
        solved = np.linalg.solve(chol.T, np.linalg.solve(chol, h @ covariance_predicted))
        gain = solved.T

        mean_filtered = mean_predicted + gain @ innovation
        residual_operator = identity - gain @ h
        covariance_filtered = (
            residual_operator @ covariance_predicted @ residual_operator.T
            + gain @ r @ gain.T
        )
        covariance_filtered = (covariance_filtered + covariance_filtered.T) / 2

        log_likelihood -= 0.5 * (
            n_yield * np.log(2.0 * np.pi)
            + 2.0 * np.log(np.diag(chol)).sum()
            + whitened @ whitened
        )
        predicted_mean[time_index] = mean_predicted
        predicted_covariance[time_index] = covariance_predicted
        filtered_mean[time_index] = mean_filtered
        filtered_covariance[time_index] = covariance_filtered
        innovations[time_index] = innovation
        innovation_covariance[time_index] = covariance_innovation
        whitened_innovations[time_index] = whitened
        mean_previous, covariance_previous = mean_filtered, covariance_filtered

    return FilterResult(
        float(log_likelihood),
        predicted_mean,
        predicted_covariance,
        filtered_mean,
        filtered_covariance,
        innovations,
        innovation_covariance,
        whitened_innovations,
    )
