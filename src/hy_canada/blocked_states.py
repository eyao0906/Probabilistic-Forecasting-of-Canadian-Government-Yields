"""Full-covariance factor filtering and backward simulation."""

import numpy as np
from numba import njit


@njit
def _draw_factors(y, loadings, mu, rho, q_time, r, m0, p0, normals):
    n = len(y)
    predicted_mean = np.empty((n, 3))
    predicted_cov = np.empty((n, 3, 3))
    filtered_mean = np.empty((n, 3))
    filtered_cov = np.empty((n, 3, 3))
    mean = m0.copy()
    cov = p0.copy()
    a = np.diag(rho)
    eye = np.eye(3)
    for t in range(n):
        if t:
            mean = mu + rho * (mean - mu)
            cov = a @ cov @ a + q_time[t - 1]
        predicted_mean[t] = mean
        predicted_cov[t] = cov
        innovation = y[t] - loadings @ mean
        innovation_cov = loadings @ cov @ loadings.T + np.diag(r)
        chol = np.linalg.cholesky(innovation_cov)
        gain = np.linalg.solve(chol.T, np.linalg.solve(chol, loadings @ cov)).T
        mean = mean + gain @ innovation
        residual_operator = eye - gain @ loadings
        cov = residual_operator @ cov @ residual_operator.T + (gain * r) @ gain.T
        cov = (cov + cov.T) * 0.5
        filtered_mean[t] = mean
        filtered_cov[t] = cov
    states = np.empty((n, 3))
    states[-1] = mean + np.linalg.cholesky(cov) @ normals[-1]
    for t in range(n - 2, -1, -1):
        gain = np.linalg.solve(predicted_cov[t + 1], a @ filtered_cov[t]).T
        mean = filtered_mean[t] + gain @ (states[t + 1] - predicted_mean[t + 1])
        cov = filtered_cov[t] - gain @ predicted_cov[t + 1] @ gain.T
        cov = (cov + cov.T) * 0.5
        states[t] = mean + np.linalg.cholesky(cov) @ normals[t]
    return states


@njit
def draw_factors(y, loadings, mu, rho, q, r, m0, p0, normals):
    """Draw factors with constant full transition covariance."""
    q_time = np.empty((len(y) - 1, 3, 3))
    for t in range(len(y) - 1):
        q_time[t] = q
    return _draw_factors(y, loadings, mu, rho, q_time, r, m0, p0, normals)


@njit
def draw_factors_sv(y, loadings, mu, rho, h, r, m0, p0, normals):
    """Draw factors with log variances h[t] for the transition into f[t+1]."""
    q_time = np.zeros((len(y) - 1, 3, 3))
    for t in range(len(y) - 1):
        for j in range(3):
            q_time[t, j, j] = np.exp(h[t, j])
    return _draw_factors(y, loadings, mu, rho, q_time, r, m0, p0, normals)
