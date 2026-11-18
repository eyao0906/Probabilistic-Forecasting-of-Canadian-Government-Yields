"""Seven-component SV augmentation and corrected blocked volatility moves."""

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import multivariate_t

WEIGHTS = np.array([0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750])
MEANS = np.array([-11.40039, -5.24321, -9.83726, 1.50746, -0.65098, 0.52478, -2.35859])
VARIANCES = np.array([5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261])


def mixture_log_probabilities(observation, h):
    logp = (np.log(WEIGHTS)[None, :]
            - 0.5 * (np.log(2 * np.pi * VARIANCES)[None, :]
                     + (observation[:, None] - h[:, None] - MEANS[None, :]) ** 2
                     / VARIANCES[None, :]))
    return logp - logsumexp(logp, axis=1, keepdims=True)


def mixture_indicators(observation, h, rng):
    probabilities = np.exp(mixture_log_probabilities(observation, h))
    cumulative = np.cumsum(probabilities, axis=1)
    cumulative[:, -1] = 1.0
    return (rng.random(len(h))[:, None] > cumulative).sum(axis=1)


@njit
def scalar_filter(observation, observation_variances, mu, phi, sigma):
    mean = mu
    variance = sigma * sigma / (1 - phi * phi)
    loglik = 0.0
    for t in range(len(observation)):
        if t:
            mean = mu + phi * (mean - mu)
            variance = phi * phi * variance + sigma * sigma
        innovation = observation[t] - mean
        s = variance + observation_variances[t]
        loglik -= 0.5 * (np.log(2 * np.pi * s) + innovation * innovation / s)
        mean += variance / s * innovation
        variance *= observation_variances[t] / s
    return loglik


@njit
def scalar_smoother(observation, observation_variances, mu, phi, sigma, normals):
    n = len(observation)
    pm, pv, fm, fv = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    mean, variance = mu, sigma * sigma / (1 - phi * phi)
    for t in range(n):
        if t:
            mean = mu + phi * (mean - mu)
            variance = phi * phi * variance + sigma * sigma
        pm[t], pv[t] = mean, variance
        s = variance + observation_variances[t]
        mean += variance / s * (observation[t] - mean)
        variance *= observation_variances[t] / s
        fm[t], fv[t] = mean, variance
    result = np.empty(n)
    result[-1] = mean + np.sqrt(variance) * normals[-1]
    for t in range(n - 2, -1, -1):
        gain = fv[t] * phi / pv[t + 1]
        mean = fm[t] + gain * (result[t + 1] - pm[t + 1])
        variance = fv[t] - gain * gain * pv[t + 1]
        result[t] = mean + np.sqrt(variance) * normals[t]
    return result


def transformed_parameters(theta):
    return theta[0], expit(theta[1]), np.exp(theta[2])


def marginal_log_target(theta, observation, indices, prior_center):
    if not np.isfinite(theta).all() or abs(theta[1]) > 25 or abs(theta[2]) > 15:
        return -np.inf
    mu, phi, sigma = transformed_parameters(theta)
    ll = scalar_filter(observation - MEANS[indices], VARIANCES[indices], mu, phi, sigma)
    prior = -0.5 * np.sum(((theta - prior_center) / np.array([2.0, 1.25, 1.0])) ** 2)
    return float(ll + prior)


def proposal_distribution(observation, indices, prior_center):
    def objective(x):
        value = marginal_log_target(x, observation, indices, prior_center)
        return -value if np.isfinite(value) else 1e100

    fit = minimize(objective, prior_center, method="BFGS", options={"gtol": 1e-4, "maxiter": 100})
    if (not np.isfinite(fit.x).all() or not np.isfinite(fit.jac).all()
            or (not fit.success and np.linalg.norm(fit.jac) > 5e-3)):
        raise RuntimeError(f"SV proposal optimization failed: {fit.message}; gradient={fit.jac}")
    covariance = np.asarray(fit.hess_inv)
    covariance = (covariance + covariance.T) / 2
    np.linalg.cholesky(covariance)
    return fit.x, covariance


def log_correction_weight(errors, h, offset):
    """Exact Gaussian / seven-component likelihood for fixed factor innovations."""
    if offset < 0:
        raise ValueError("Offset must be nonnegative")
    squared = errors * errors + offset
    if np.any(squared <= 0):
        raise ValueError("Zero innovation with zero offset has undefined log transform")
    transformed = np.log(squared)
    log_components = (np.log(WEIGHTS)[None, :]
                      - 0.5 * (np.log(2 * np.pi * VARIANCES)[None, :]
                               + (transformed[:, None] - h[:, None] - MEANS[None, :])**2
                               / VARIANCES[None, :]))
    approx = logsumexp(log_components, axis=1)
    exact = -0.5 * (np.log(2 * np.pi) + h + errors**2 * np.exp(-h))
    return float(np.sum(exact - approx))


def parameter_log_acceptance(current, candidate, observation, indices, prior_center, center, scale):
    return (marginal_log_target(candidate, observation, indices, prior_center)
            - marginal_log_target(current, observation, indices, prior_center)
            + multivariate_t.logpdf(current, loc=center, shape=scale, df=5)
            - multivariate_t.logpdf(candidate, loc=center, shape=scale, df=5))


def volatility_block(errors, h, theta, prior_center, rng, offset=0.001):
    if offset < 0 or np.any(errors * errors + offset <= 0):
        raise ValueError("SV log-squared innovations require a nonnegative offset and positive argument")
    observation = np.log(errors * errors + offset)
    indices = mixture_indicators(observation, h, rng)
    center, scale = proposal_distribution(observation, indices, prior_center)
    candidate = center + np.linalg.cholesky(scale) @ rng.standard_normal(3) / np.sqrt(rng.chisquare(5) / 5)
    ratio = parameter_log_acceptance(theta, candidate, observation, indices,
                                     prior_center, center, scale)
    theta_accept = np.log(rng.random()) < ratio
    next_theta = candidate if theta_accept else theta
    mu, phi, sigma = transformed_parameters(next_theta)
    next_h = scalar_smoother(observation - MEANS[indices], VARIANCES[indices],
                             mu, phi, sigma, rng.standard_normal(len(h)))
    old_weight = log_correction_weight(errors, h, offset)
    new_weight = log_correction_weight(errors, next_h, offset)
    corrected = np.log(rng.random()) < new_weight - old_weight
    if corrected:
        return next_h, next_theta, bool(theta_accept), True, new_weight - old_weight
    return h, theta, bool(theta_accept), False, new_weight - old_weight
