"""Hautsch-Yang-style blocked Gibbs sampler for joint Canadian DLNS."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.special import logsumexp
from scipy.stats import invwishart, truncnorm

from .blocked_states import draw_factors
from .data import MATURITIES
from .kalman import loadings

PARAMETER_NAMES = (
    [f"mu_{j}" for j in "LSC"] + [f"rho_{j}" for j in "LSC"]
    + ["q_LL", "q_SS", "q_CC", "q_LS", "q_LC", "q_SC"]
    + [f"r_{m:g}" for m in MATURITIES] + ["decay"]
)


@dataclass(frozen=True)
class Priors:
    m0: np.ndarray
    p0: np.ndarray
    mu_mean: np.ndarray
    mu_variance: np.ndarray
    rho_mean: float
    rho_variance: float
    r_shape: float
    r_scale: np.ndarray
    q_df: float
    q_scale: np.ndarray


def training_priors(y):
    h = loadings(MATURITIES, 0.7)
    factors = np.linalg.lstsq(h, y.T, rcond=None)[0].T
    mean = factors.mean(axis=0)
    sd = np.maximum(factors.std(axis=0, ddof=1), 0.1)
    residuals = y - factors @ h.T
    centered = factors - mean
    rho = np.sum(centered[1:] * centered[:-1], axis=0) / np.sum(centered[:-1] ** 2, axis=0)
    rho = np.clip(rho, 0.7, 0.995)
    innovations = centered[1:] - rho * centered[:-1]
    q_scale = np.cov(innovations.T) + np.eye(3) * 1e-5
    r0 = np.maximum(residuals.var(axis=0, ddof=1), 1e-5)
    return Priors(mean, np.diag((2 * sd) ** 2), mean, (2 * sd) ** 2,
                  0.95, 0.1**2, 2.5, 1.5 * r0, 5.0, q_scale)


def measurement_conditional(y, factors, h, prior):
    residuals = y - factors @ h.T
    return prior.r_shape + len(y) / 2, prior.r_scale + np.sum(residuals**2, axis=0) / 2


def decay_probabilities(y, factors, r, grid, grid_loadings=None):
    if np.any(np.diff(grid) <= 0) or np.any(grid <= 0):
        raise ValueError("Decay grid must be increasing and positive")
    if grid_loadings is None:
        grid_loadings = np.array([loadings(MATURITIES, decay) for decay in grid])
    cross = y.T @ factors
    gram = factors.T @ factors
    logp = (np.einsum("gij,ij,i->g", grid_loadings, cross, 1 / r)
            - 0.5 * np.einsum("gij,jk,gik,i->g", grid_loadings, gram, grid_loadings, 1 / r))
    return np.exp(logp - logsumexp(logp))


def dynamics_conditionals(factors, mu, rho, q, prior, j):
    """Normal parameters for mu_j and rho_j, conditional on other components."""
    precision = np.linalg.inv(q)
    previous, current = factors[:-1], factors[1:]
    residuals = current - mu - rho * (previous - mu)
    cross = residuals @ precision[j] - residuals[:, j] * precision[j, j]
    cross /= precision[j, j]
    coeff_mu = 1 - rho[j]
    mu_var = 1 / (len(previous) * precision[j, j] * coeff_mu**2 + 1 / prior.mu_variance[j])
    mu_mean = mu_var * (precision[j, j] * coeff_mu
                        * np.sum(current[:, j] - rho[j] * previous[:, j] + cross)
                        + prior.mu_mean[j] / prior.mu_variance[j])
    x = previous[:, j] - mu[j]
    rho_var = 1 / (precision[j, j] * (x @ x) + 1 / prior.rho_variance)
    rho_mean = rho_var * (precision[j, j] * (x @ (current[:, j] - mu[j] + cross))
                          + prior.rho_mean / prior.rho_variance)
    return mu_mean, mu_var, rho_mean, rho_var


def innovation_conditional(factors, mu, rho, prior):
    residuals = factors[1:] - mu - rho * (factors[:-1] - mu)
    return prior.q_df + len(residuals), prior.q_scale + residuals.T @ residuals


def sample_chain(y, prior, seed, warmup, draws, grid, keep_paths=20, initial_decay=0.7):
    if warmup < 0 or draws < 1 or keep_paths < 1 or keep_paths > draws:
        raise ValueError("Invalid chain lengths or number of stored factor paths")
    rng = np.random.default_rng(seed)
    decay = float(grid[np.argmin(np.abs(grid - initial_decay))])
    factors = np.linalg.lstsq(loadings(MATURITIES, decay), y.T, rcond=None)[0].T
    mu = prior.mu_mean + rng.normal(size=3) * np.sqrt(prior.mu_variance) * 0.1
    rho = np.clip(rng.normal(0.94, 0.025, size=3), 0.8, 0.995)
    q = prior.q_scale / (prior.q_df - 4)
    r = prior.r_scale / (prior.r_shape - 1)
    grid_loadings = np.array([loadings(MATURITIES, value) for value in grid])
    output = np.empty((draws, len(PARAMETER_NAMES)))
    paths = np.empty((keep_paths, len(y), 3), dtype=np.float32)
    path_indices = np.linspace(0, draws - 1, keep_paths, dtype=int)
    path_index = 0
    start = perf_counter()
    for iteration in range(warmup + draws):
        h = loadings(MATURITIES, decay)
        shape, scale = measurement_conditional(y, factors, h, prior)
        r = 1 / rng.gamma(shape, 1 / scale)
        probabilities = decay_probabilities(y, factors, r, grid, grid_loadings)
        decay = float(rng.choice(grid, p=probabilities))
        factors = draw_factors(y, loadings(MATURITIES, decay), mu, rho, q, r,
                               prior.m0, prior.p0, rng.standard_normal((len(y), 3)))
        for j in range(3):
            mean, variance, _, _ = dynamics_conditionals(factors, mu, rho, q, prior, j)
            mu[j] = rng.normal(mean, np.sqrt(variance))
            _, _, mean, variance = dynamics_conditionals(factors, mu, rho, q, prior, j)
            sd = np.sqrt(variance)
            rho[j] = truncnorm.rvs((-1 - mean) / sd, (1 - mean) / sd,
                                   loc=mean, scale=sd, random_state=rng)
        df, scale = innovation_conditional(factors, mu, rho, prior)
        q = invwishart.rvs(df=df, scale=scale, random_state=rng)
        if iteration >= warmup:
            index = iteration - warmup
            output[index] = np.r_[mu, rho, q[0, 0], q[1, 1], q[2, 2],
                                  q[0, 1], q[0, 2], q[1, 2], r, decay]
            if path_index < keep_paths and index == path_indices[path_index]:
                paths[path_index] = factors
                path_index += 1
    return {"draws": output, "paths": paths, "path_indices": path_indices,
            "elapsed_seconds": perf_counter() - start, "seed": seed}
