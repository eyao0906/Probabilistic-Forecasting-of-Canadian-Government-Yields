"""Blocked SVNS sampler with an exact-corrected mixture volatility kernel."""

from time import perf_counter

import numpy as np
from scipy.special import logit
from scipy.stats import truncnorm

from .blocked_dlns import decay_probabilities, measurement_conditional
from .blocked_states import draw_factors_sv
from .blocked_sv import transformed_parameters, volatility_block
from .data import MATURITIES
from .kalman import loadings

PARAMETER_NAMES = ([f"mu_{j}" for j in "LSC"] + [f"rho_{j}" for j in "LSC"]
                   + [f"r_{m:g}" for m in MATURITIES] + ["decay"]
                   + [f"h_mean_{j}" for j in "LSC"]
                   + [f"h_phi_{j}" for j in "LSC"] + [f"h_sigma_{j}" for j in "LSC"])


def dynamics_conditionals(factors, mu, rho, h, prior, j):
    precision = np.exp(-h[:, j])
    previous, current = factors[:-1, j], factors[1:, j]
    coefficient = 1 - rho[j]
    mu_var = 1 / (np.sum(precision) * coefficient**2 + 1 / prior.mu_variance[j])
    mu_mean = mu_var * (coefficient * np.sum(precision * (current - rho[j] * previous))
                        + prior.mu_mean[j] / prior.mu_variance[j])
    x = previous - mu[j]
    rho_var = 1 / (np.sum(precision * x**2) + 1 / prior.rho_variance)
    rho_mean = rho_var * (np.sum(precision * x * (current - mu[j]))
                          + prior.rho_mean / prior.rho_variance)
    return mu_mean, mu_var, rho_mean, rho_var


def sample_chain(y, prior, seed, warmup, draws, grid, keep_paths=20, initial_decay=0.7,
                 offset=0.001, inner_steps=1):
    if warmup < 0 or draws < 1 or keep_paths < 1 or keep_paths > draws or inner_steps < 1:
        raise ValueError("Invalid chain lengths or number of stored paths")
    rng = np.random.default_rng(seed)
    decay = float(grid[np.argmin(np.abs(grid - initial_decay))])
    factors = np.linalg.lstsq(loadings(MATURITIES, decay), y.T, rcond=None)[0].T
    mu = prior.mu_mean + rng.normal(size=3) * np.sqrt(prior.mu_variance) * 0.1
    rho = np.clip(rng.normal(0.94, 0.025, size=3), 0.8, 0.995)
    r = prior.r_scale / (prior.r_shape - 1)
    prior_centers = np.column_stack((np.log(np.diag(prior.q_scale)),
                                     np.full(3, logit(0.95)), np.full(3, np.log(0.2))))
    theta = prior_centers.copy()
    h = np.broadcast_to(theta[:, 0], (len(y) - 1, 3)).copy()
    grid_loadings = np.array([loadings(MATURITIES, value) for value in grid])
    output = np.empty((draws, len(PARAMETER_NAMES)))
    paths = np.empty((keep_paths, len(y), 3), dtype=np.float32)
    volatility_paths = np.empty((keep_paths, len(y) - 1, 3), dtype=np.float32)
    path_indices = np.linspace(0, draws - 1, keep_paths, dtype=int)
    theta_accepts = np.zeros(3, dtype=int)
    corrections = np.zeros(3, dtype=int)
    logratios = np.empty((draws, 3))
    path_index = 0
    start = perf_counter()
    for iteration in range(warmup + draws):
        shape, scale = measurement_conditional(y, factors, loadings(MATURITIES, decay), prior)
        r = 1 / rng.gamma(shape, 1 / scale)
        probabilities = decay_probabilities(y, factors, r, grid, grid_loadings)
        decay = float(rng.choice(grid, p=probabilities))
        factors = draw_factors_sv(y, loadings(MATURITIES, decay), mu, rho, h, r,
                                  prior.m0, prior.p0, rng.standard_normal((len(y), 3)))
        for j in range(3):
            mean, variance, _, _ = dynamics_conditionals(factors, mu, rho, h, prior, j)
            mu[j] = rng.normal(mean, np.sqrt(variance))
            _, _, mean, variance = dynamics_conditionals(factors, mu, rho, h, prior, j)
            sd = np.sqrt(variance)
            rho[j] = truncnorm.rvs((-1 - mean) / sd, (1 - mean) / sd,
                                   loc=mean, scale=sd, random_state=rng)
        errors = factors[1:] - mu - rho * (factors[:-1] - mu)
        for j in range(3):
            for _ in range(inner_steps):
                h[:, j], theta[j], theta_accepted, corrected, logratio = volatility_block(
                    errors[:, j], h[:, j], theta[j], prior_centers[j], rng, offset)
                if iteration >= warmup:
                    theta_accepts[j] += theta_accepted
                    corrections[j] += corrected
            if iteration >= warmup:
                logratios[iteration - warmup, j] = logratio
        if iteration >= warmup:
            index = iteration - warmup
            parameters = np.array([transformed_parameters(row) for row in theta])
            output[index] = np.r_[mu, rho, r, decay, parameters[:, 0],
                                  parameters[:, 1], parameters[:, 2]]
            if path_index < keep_paths and index == path_indices[path_index]:
                paths[path_index] = factors
                volatility_paths[path_index] = h
                path_index += 1
    return {"draws": output, "paths": paths, "volatility_paths": volatility_paths,
            "path_indices": path_indices, "elapsed_seconds": perf_counter() - start,
            "theta_accepts": theta_accepts, "corrections": corrections,
            "logratios": logratios, "seed": seed}
