"""Non-centered SVNS inference with collapsed factor states."""

import time

import numpy as np
from numba import njit
from scipy.special import expit, logit

from .data import MATURITIES
from .dlns import BLOCKS, TrainingPrior, parameters
from .kalman import filter_likelihood, loadings

SV_NAMES = [f"sv_phi_{x}" for x in "LSC"] + [f"sv_variance_{x}" for x in "LSC"]
SV_BLOCKS = BLOCKS + (slice(18, 21), slice(21, 24))


@njit
def log_variances(innovations, phi, variance, log_mean_q):
    """Map independent standard normals to stationary AR log variances."""
    n = len(innovations)
    h = np.empty((n, 3))
    for j in range(3):
        state = innovations[0, j]
        h[0, j] = log_mean_q[j] - variance[j] / 2 + np.sqrt(variance[j]) * state
        scale = np.sqrt(1 - phi[j] ** 2)
        for t in range(1, n):
            state = phi[j] * state + scale * innovations[t, j]
            h[t, j] = log_mean_q[j] - variance[j] / 2 + np.sqrt(variance[j]) * state
    return h


def sv_prior(prior: TrainingPrior) -> tuple[np.ndarray, np.ndarray]:
    return (np.r_[prior.center, np.full(3, logit(0.95)), np.full(3, np.log(0.5))],
            np.r_[prior.scale, np.full(3, 1.25), np.ones(3)])


def sv_likelihood(z, noise, y, prior):
    if not np.isfinite(z).all() or np.any(np.abs(z[6:17]) > 50) or np.any(np.abs(z[21:24]) > 10):
        return -np.inf
    mu, rho, _, r, decay = parameters(z)
    h = log_variances(noise, expit(z[18:21]), np.exp(z[21:24]), z[6:9])
    if not np.isfinite(h).all() or np.any(np.abs(h) > 50):
        return -np.inf
    return filter_likelihood(y, loadings(MATURITIES, decay), mu, rho, np.exp(h), r,
                             prior.initial_mean, prior.initial_covariance)


def sv_log_posterior(z, noise, y, prior):
    center, scale = sv_prior(prior)
    return float(sv_likelihood(z, noise, y, prior)
                 - 0.5 * np.sum(((z - center) / scale) ** 2)
                 - 0.5 * np.sum(noise**2))


def elliptical_slice(path, current_ll, evaluate, rng):
    """Update a standard-normal prior block using its conditional likelihood."""
    direction = rng.standard_normal(path.shape)
    threshold = current_ll + np.log(rng.random())
    angle = rng.uniform(0, 2 * np.pi)
    lower, upper = angle - 2 * np.pi, angle
    for evaluations in range(1, 1001):
        proposal = path * np.cos(angle) + direction * np.sin(angle)
        next_ll = evaluate(proposal)
        if next_ll > threshold:
            return proposal, next_ll, evaluations
        if angle < 0:
            lower = angle
        else:
            upper = angle
        angle = rng.uniform(lower, upper)
    raise RuntimeError("Elliptical slice failed to find an acceptable path")


def sample_chain(y, prior, seed, warmup, draws):
    if warmup < 0 or draws < 1:
        raise ValueError("Warmup must be nonnegative and draws positive")
    rng = np.random.default_rng(seed)
    center, prior_scale = sv_prior(prior)
    z = center + rng.normal(size=24) * prior_scale * 0.15
    noise = rng.standard_normal((len(y), 3))
    ll = sv_likelihood(z, noise, y, prior)
    lp = -0.5 * np.sum(((z - center) / prior_scale) ** 2)
    if not np.isfinite(ll + lp):
        raise ValueError("Nonfinite initial SVNS log posterior")
    scales = np.array([0.10, 0.16, 0.08, 0.06, 0.04, 0.14, 0.12])
    accepted = np.zeros(7, dtype=int)
    window = np.zeros(7, dtype=int)
    evaluations = np.zeros(3, dtype=int)
    stored = np.empty((draws, 24))
    paths = np.empty((draws, len(y), 3), dtype=np.float32)
    start = time.perf_counter()
    for iteration in range(warmup + draws):
        for i, block in enumerate(SV_BLOCKS):
            proposal = z.copy()
            proposal[block] += rng.normal(size=prior_scale[block].size) * prior_scale[block] * scales[i]
            next_ll = sv_likelihood(proposal, noise, y, prior)
            next_lp = -0.5 * np.sum(((proposal - center) / prior_scale) ** 2)
            if np.log(rng.random()) < next_ll + next_lp - ll - lp:
                z, ll, lp = proposal, next_ll, next_lp
                if iteration >= warmup:
                    accepted[i] += 1
                else:
                    window[i] += 1
        for j in range(3):
            def evaluate(path):
                proposal_noise = noise.copy()
                proposal_noise[:, j] = path
                return sv_likelihood(z, proposal_noise, y, prior)

            path, ll, used = elliptical_slice(noise[:, j], ll, evaluate, rng)
            noise[:, j] = path
            if iteration >= warmup:
                evaluations[j] += used
        if iteration < warmup and (iteration + 1) % 50 == 0:
            scales *= np.exp(np.clip(window / 50 - 0.25, -0.25, 0.25))
            window[:] = 0
        if iteration >= warmup:
            mu, rho, q, r, decay = parameters(z)
            stored[iteration - warmup] = np.r_[mu, rho, q, r, decay,
                                                expit(z[18:21]), np.exp(z[21:24])]
            paths[iteration - warmup] = log_variances(noise, expit(z[18:21]),
                                                       np.exp(z[21:24]), z[6:9])
    return {"draws": stored, "paths": paths, "acceptance": accepted / draws,
            "ellipse_evaluations_per_update": evaluations / draws, "scales": scales,
            "elapsed_seconds": time.perf_counter() - start, "seed": seed}
